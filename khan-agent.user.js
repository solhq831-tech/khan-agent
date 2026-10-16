// ==UserScript==
// @name         Khan Agent (Pollinations) — Assistente Autonomo
// @namespace    https://github.com/local/khan-agent
// @version      1.6.0
// @description  Assistente autonomo que le o Khan Academy, consulta uma IA OpenAI-compativel e responde sozinho: multipla escolha, teclado, Khan exercises e vision.
// @match        *://*.khanacademy.org/*
// @match        *://*.khanacademy.com/*
// @run-at       document-idle
//
// GRANT: NAO escreva "@grant none" aqui.
//
// "@grant none" manda o Tampermonkey rodar o script no contexto da pagina SEM
// nenhuma API GM -- e ai o GM_xmlhttpRequest nao existe, que e exatamente o
// que o agente precisa para falar com provider que nao libera CORS (NVIDIA,
// por exemplo). Declarar os dois e o pior dos casos: o gerenciador ve "none",
// nao fornece a API, e o script morre em "Failed to fetch" sem explicar.
//
// Com "@grant GM_xmlhttpRequest" o script roda no sandbox do Tampermonkey. O
// DOM continua sendo o do Khan (isolated world compartilha o DOM), entao
// clicar, digitar e ler o enunciado funcionam igual.
//
// CONNECT: o endpoint e configuravel pelo usuario, entao a lista nao pode ser
// fechada -- faltando o host, o Tampermonkey bloqueia a requisicao com um
// erro que nao parece de CORS. Os hosts conhecidos ficam declarados por
// clareza, e o "*" cobre proxy e provider novo.
//
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @connect      gen.pollinations.ai
// @connect      text.pollinations.ai
// @connect      api.groq.com
// @connect      api.openai.com
// @connect      openrouter.ai
// @connect      ai.api.nvidia.com
// @connect      integrate.api.nvidia.com
// @connect      api.cohere.com
// @connect      api.together.xyz
// @connect      api.mistral.ai
// @connect      localhost
// @connect      127.0.0.1
// @connect      *
// ==/UserScript==

/* =============================================================================
 * KHAN AGENT
 * -----------------------------------------------------------------------------
 * Agente autonomo que age como o usuario dentro do Khan Academy.
 *
 * Ciclo:
 *   1. Observa o DOM e espera aparecer um exercicio sem resposta.
 *   2. Extrai enunciado + alternativas + campos de resposta.
 *   3. Monta um catalogo de acoes possiveis e envia o contexto para a IA.
 *   4. A IA responde UM objeto JSON ("plan"). A IA decide a LOGICA.
 *   5. O script executa a MECANICA de interface (marcar, digitar, montar,
 *      checar, avancar). A IA nunca envia coordenadas brutas: so escolhe
 *      indices do catalogo que mandamos. Isso mantem o agente no trilho.
 *   6. Repete quando a proxima questao carrega.
 *
 * Apoia: multipla escolha (radios/checkboxes), resposta livre (inputs de texto),
 *        Khan exercises / expressoes em LaTeX, e navegacao (check, next, hint).
 * ========================================================================== */

(() => {
  'use strict';

  /* ===========================================================================
   * CONFIGURACAO
   * ======================================================================== */
  const CFG = {
    // Host novo e recomendado. Sem API key o script cai para o host legado.
    endpoint: 'https://gen.pollinations.ai/v1/chat/completions',
    fallbackEndpoint: 'https://text.pollinations.ai/openai',

    model: 'openai',          // qualquer modelId valido em /v1/models
    apiKey: '',               // opcional: sk_... ou pk_... de enter.pollinations.ai

    // Espera entre observacoes. Era 1500 ms, e a maior parte disso era
    // desperdicio: waitSettle() ja garante que a pagina assentou antes do loop
    // decidir de novo, entao 800 ms basta para nao martelar o React.
    // Configuravel -- quem usa um provider rapido pode baixar, quem usa um
    // lento pode subir.
    loopDelayMs: 800,
    maxRetries: 3,            // tentativas por endpoint
    requestTimeoutMs: 60000,  // timeout por requisicao
    maxProblemChars: 3500,    // trunca enunciados gigantes
    memoryTurns: 6,           // trocas de memoria mantidas no historico do agente
    maxAttemptsPerQuestion: 2, // evita loop infinito em questao sem solucao
    maxOutputTokens: 2000,     // teto de tokens da resposta da IA (ver askAI)
    humanDelay: [90, 260],    // faixa de delay "humano" entre acoes (ms)
    autoCheck: true,          // clica em "Check answer" depois de responder
    vision: true,             // anexa o grafico/imagem da questao a mensagem da IA
    // Tool calling nativo (campo `tools` no request). Quando o provider recusa,
    // o script cai sozinho para o JSON no texto -- ver askAI.
    toolCalling: true,
    maxPlanSteps: 6,          // teto de passos de um plano {"actions":[...]}

    // Backend de chamada. 'http' = endpoint OpenAI-compativel (com Bearer key).
    // 'puter' = SDK do Puter no navegador: sem chave nenhuma, autenticado pela
    // conta do usuario, e sem CORS porque e feito para rodar no cliente.
    backend: 'http',
  };

  const STORAGE_KEY = 'khan_agent_cfg_v1';

  // Quantas observacoes IDENTICAS seguidas a pagina pode devolver antes de o
  // agente desistir. Tres e suficiente para cobrir "o React re-renderiza" e
  // "o Check ainda esta processando"; a partir da quarta, clicar e so gastar
  // cota da API sem mudar nada.
  const STALL_LIMIT = 3;

  /* ===========================================================================
   * ESTADO
   * ======================================================================== */
  const state = {
    running: false,
    busy: false,
    snapshot: null,
    history: [],             // [{ role, content }] memoria curta
    logLines: [],            // log do painel
    solved: 0,
    wrong: 0,
    epoch: 0,                // muda a cada navegacao -> invalida dedupe
    seen: new Map(),         // fingerprint -> tentativas
    lastFingerprint: null,
    stuckAdvances: 0,        // Avancadas seguidas sem troca de questao
    stuckFingerprint: null,
    stallTicks: 0,           // Observacoes identicas seguidas (trava global)
    stallFingerprint: null,
    lastProgress: null,      // Ultimo contador de progresso visto na pagina
    visionDead: false,       // Provider/modelo recusou imagem nesta sessao
    aguardandoUsuario: false, // ask_user parou o loop esperando resposta no chat
    visionWhy: '',           // Motivo da ultima falha de captura
    imgStyle: null,          // Formato da parte de imagem: null | 'string'
    singleMessage: false,    // Endpoint aceita so 1 mensagem (sem role system)
    toolsDead: false,        // Endpoint/modelo recusou o campo `tools` nesta sessao
    toolChoiceDead: false,   // Aceita `tools` mas recusou `tool_choice`
    extrasDead: false,       // Endpoint aceita so UMA imagem: sem alternativas-imagem
    useGm: false,            // Provider barrou CORS: ir por GM_xmlhttpRequest
    // Memoria do agente NA QUESTAO ATUAL (ver registrarTentativa). Zerada
    // quando a chave da questao muda -- ver chaveDaQuestao/decide.
    plano: { chave: null, tentativas: [] },
    // Catalogo de modelos do endpoint (ver buscarModelos). Nao e config: por
    // isso fica fora de state.cfg e nao vai para o saveCfg.
    modelos: [],
    // Chave da ultima questao diagnosticada sem controles, para o aviso nao
    // repetir a cada tique do loop (ver diagnosticarSemControles).
    diagChave: null,
    // O que o modelo DISSE na ultima resposta (content + campos de raciocinio).
    // Ver extrairNarracao: e o que permite saber o que ele pretendia quando a
    // resposta vem por tool calling, sem texto.
    narracao: '',
    // Fingerprint em que o agente JA AGIU. Separa "o React re-renderizou" de
    // "eu agi e a pagina nao mudou" (calculate, wait) -- ver o dedupe no loop.
    agiuFingerprint: null,
    cfg: { ...CFG },
  };

  /* ===========================================================================
   * UTILIDADES
   * ======================================================================== */
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const humanDelay = () =>
    sleep(state.cfg.humanDelay[0] + Math.random() * (state.cfg.humanDelay[1] - state.cfg.humanDelay[0]));

  function loadCfg() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) Object.assign(state.cfg, JSON.parse(raw));
    } catch { /* ignora storage bloqueado */ }
    // Migracao do default antigo de loopDelayMs (1500 -> 800).
    //
    // Quem nunca mexeu no campo tem 1500 salvo no localStorage, e o novo
    // default nao chegaria sem isto. O condicional e exatamente "era o valor
    // padrao": se o usuario TROCOU para 1500 de proposito, a escolha dele foi
    // outra e o valor dela e respeitado.
    if (state.cfg.loopDelayMs === 1500) state.cfg.loopDelayMs = CFG.loopDelayMs;
  }
  function saveCfg() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.cfg)); } catch { /* noop */ }
  }

  const log = (msg, kind = 'info') => {
    state.logLines.push({ msg, kind, at: new Date().toLocaleTimeString('pt-BR', { hour12: false }) });
    if (state.logLines.length > 500) state.logLines.shift();
    paintLog();
  };

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
    );
  }

  function hash(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) | 0;
    return (h >>> 0).toString(36);
  }

  /* ===========================================================================
   * CALCULADORA (ferramenta `calculate`)
   * ---------------------------------------------------------------------------
   * A IA e boa em algebra e ruim em aritmetica. Esta funcao existe para o modelo
   * NAO fazer conta de cabeca: ele manda a expressao e usa o numero que voltar.
   *
   * Nao usa eval() nem new Function(): o texto vem de uma resposta de IA, e
   * avaliar codigo arbitrario dentro da pagina do usuario nao se justifica quando
   * um parser de 60 linhas resolve. `eval` aqui seria uma porta de execucao
   * aberta por um dado que nao controlamos.
   *
   * Gramatica, do mais frouxo ao mais justo:
   *   expr     := termo (('+'|'-') termo)*
   *   termo    := unario (('*'|'/'|'%') unario)*
   *   unario   := ('-'|'+') unario | potencia
   *   potencia := primario ('^' unario)?      -- associativa a direita
   *   primario := numero | func '(' args ')' | const | var | '(' expr ')'
   *
   * `unario` acima de `potencia` e o que faz "-2^2" dar -4 (e nao 4), que e a
   * convencao matematica. Com a ordem invertida, o sinal entraria na base.
   * ======================================================================== */
  const CALC_FUNCOES = {
    sqrt: Math.sqrt, abs: Math.abs, round: Math.round, floor: Math.floor,
    ceil: Math.ceil, log: Math.log10, ln: Math.log, exp: Math.exp,
    sin: Math.sin, cos: Math.cos, tan: Math.tan, min: Math.min, max: Math.max,
  };
  const CALC_CONSTANTES = { pi: Math.PI, e: Math.E };

  function calcular(expressao, vars) {
    const fonte = String(expressao ?? '');
    const variaveis = vars && typeof vars === 'object' ? vars : {};
    let i = 0;

    const pular = () => { while (i < fonte.length && /\s/.test(fonte[i])) i++; };
    const erro = (m) => { throw new Error(m); };

    const numero = () => {
      pular();
      const ini = i;
      while (i < fonte.length && /[0-9.]/.test(fonte[i])) i++;
      if (i === ini) return null;
      const v = Number(fonte.slice(ini, i));
      if (!Number.isFinite(v)) erro(`numero invalido "${fonte.slice(ini, i)}"`);
      return v;
    };

    const identificador = () => {
      pular();
      const ini = i;
      while (i < fonte.length && /[A-Za-z_]/.test(fonte[i])) i++;
      return i > ini ? fonte.slice(ini, i) : null;
    };

    function parseExpr() {
      let v = parseTermo();
      for (;;) {
        pular();
        if (fonte[i] === '+') { i++; v += parseTermo(); }
        else if (fonte[i] === '-') { i++; v -= parseTermo(); }
        else return v;
      }
    }

    function parseTermo() {
      let v = parseUnario();
      for (;;) {
        pular();
        if (fonte[i] === '*') { i++; v *= parseUnario(); }
        else if (fonte[i] === '/') { i++; v /= parseUnario(); }
        else if (fonte[i] === '%') { i++; v %= parseUnario(); }
        else return v;
      }
    }

    function parseUnario() {
      pular();
      if (fonte[i] === '-') { i++; return -parseUnario(); }
      if (fonte[i] === '+') { i++; return parseUnario(); }
      return parsePotencia();
    }

    function parsePotencia() {
      const base = parsePrimario();
      pular();
      if (fonte[i] === '^') { i++; return Math.pow(base, parseUnario()); }
      return base;
    }

    function parsePrimario() {
      pular();
      if (fonte[i] === '(') {
        i++;
        const v = parseExpr();
        pular();
        if (fonte[i] !== ')') erro('falta fechar um parenteses');
        i++;
        return v;
      }
      const n = numero();
      if (n !== null) return n;
      const nome = identificador();
      if (!nome) erro(`nao entendi a partir de "${fonte.slice(i, i + 12)}"`);
      pular();
      if (fonte[i] === '(') {
        i++;
        const args = [];
        pular();
        if (fonte[i] !== ')') {
          for (;;) {
            args.push(parseExpr());
            pular();
            if (fonte[i] === ',') { i++; continue; }
            break;
          }
        }
        pular();
        if (fonte[i] !== ')') erro('falta fechar o parenteses da funcao');
        i++;
        const f = CALC_FUNCOES[nome.toLowerCase()];
        if (!f) erro(`funcao desconhecida "${nome}"`);
        return f(...args);
      }
      const chave = nome.toLowerCase();
      if (chave in CALC_CONSTANTES) return CALC_CONSTANTES[chave];
      const valor = variaveis[nome] ?? variaveis[chave];
      if (typeof valor === 'number' && Number.isFinite(valor)) return valor;
      erro(`valor desconhecido para "${nome}" (passe em vars)`);
      return 0;
    }

    try {
      const valor = parseExpr();
      pular();
      if (i < fonte.length) erro(`sobrou "${fonte.slice(i, i + 12)}" na expressao`);
      if (!Number.isFinite(valor)) erro('o resultado nao e um numero finito');
      // Corta o ruido de ponto flutuante (0.30000000000000004). 12 digitos
      // significativos preservam qualquer resposta escolar e limpam o resto.
      const limpo = Math.abs(valor) < 1e-12 ? 0 : Number(valor.toPrecision(12));
      return { valor: limpo, texto: String(limpo) };
    } catch (e) {
      return { erro: e.message };
    }
  }

  // Texto da dica aberta no Khan. A acao `hint` clica no botao, mas antes disso
  // o clique era desperdicado: nada do que o Khan respondia voltava para a IA,
  // entao pedir dica nao ajudava em nada.
  //
  // Tolerante de proposito -- o Khan troca o container entre builds. Se nenhum
  // seletor casar, devolve vazio e o log diz que nao achou, em vez de mentir.
  function lerDica() {
    const seletores = [
      '[data-testid="hint-content"]',
      '[data-testid*="hint"]',
      '.perseus-hint',
      '.hint-content',
      '.hints-and-solutions .perseus-renderer',
      '.hints-and-solutions',
    ];
    for (const sel of seletores) {
      for (const el of document.querySelectorAll(sel)) {
        if (!isVisible(el)) continue;
        const t = (el.innerText || '').replace(/\s+/g, ' ').trim();
        if (t.length > 4) return t.slice(0, 600);
      }
    }
    return '';
  }

  /* ===========================================================================
   * CLIENTE DA IA (Pollinations)
   * ======================================================================== */

  // Extrai JSON de uma resposta possivelmente embrulhada em ```json ... ```
  function parseJsonLoose(text) {
    if (!text) throw new Error('resposta vazia');
    let raw = text.trim();
    const fence = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
    if (fence) raw = fence[1].trim();
    try { return JSON.parse(raw); } catch { /* continua */ }
    // Ultimo recurso: primeiro objeto balanceado.
    const start = raw.indexOf('{');
    if (start === -1) throw new Error('sem JSON na resposta');
    let depth = 0, inStr = false, esc = false;
    for (let i = start; i < raw.length; i++) {
      const c = raw[i];
      if (inStr) {
        if (esc) esc = false;
        else if (c === '\\') esc = true;
        else if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') inStr = true;
      else if (c === '{') depth++;
      else if (c === '}') { depth--; if (depth === 0) return JSON.parse(raw.slice(start, i + 1)); }
    }
    throw new Error('JSON nao balanceado');
  }

  // Texto da resposta, seja qual for o formato da API.
  //
  // Tres formatos convolem no mesmo endpoint "/chat":
  //   OpenAI/Groq/Mistral: choices[0].message.content como STRING.
  //   Cohere v2:           message.content como LISTA de partes, e a primeira
  //                        parte costuma ser {type:"thinking"} -- o modelo
  //                        pensando alto, que NAO e a resposta.
  //   Anthropic:           content[] com {type:"text"}.
  //
  // Ler so choices[0] com o Cohere dava "resposta sem texto" mesmo com a API
  //respondendo 200 -- e, pior, engoliria a parte de raciocinio junto se
  // fosse incluido por accident, fazendo o agente lidar com o pensamento
  // interno do modelo como se fosse o plano.
  /* ===========================================================================
   * BACKEND PUTER (puter.js)
   * ======================================================================== */
  // Puter e o unico backend que resolve de uma vez tres problemas quehoy
  // custaram workaround: nao tem API key para colar, nao tem CORS para
  // negociar (o SDK e feito para rodar no navegador), e tem modelo de visao
  // (gemini-3.8-flash, gpt-5.6, claude) sem depender de plano free de ninguem.
  //
  // Como o SDK e carregado: <script src="https://js.puter.com/v2/">. Nao da para
  // usar @require num userscript de forma confiavel (o gerenciador nem sempre
  // injeta), entao carregamos por conta via <script> e esperamos o global.
  //
  // Duas limitacoes da API que moldaram o codigo:
  //
  // 1) A imagem de entrada vai pelo ARGUMENTO media, nao dentro de messages.
  //    A documentacao de entrada lista partes do tipo "text" e "file"
  //    (puter_path); "image_url" so aparece na SAIDA (message.images) e no
  //    exemplo de edicao multi-turno. O `media` aceita string de URL ou um
  //    objeto File -- e um File dá para construir a partir do data URL que o
  //    agente ja tem, sem passar pela rede. Passar o data URL como se fosse
  //    "URL" nao e garantido pela API, entao usamos File.
  //
  // 2) Nao existe overload de chat(messages, media, ...): media so combina com
  //    prompt em string. Como o prompt de sistema entao nao cabe separado,
  //    usamos o mesmo dobramento do modo "uma mensagem so" (singleMessage),
  //    que ja existe por causa do NVIDIA.
  const PUTER_SDK = 'https://js.puter.com/v2/';
  let puterPromise = null;

  // Onde o SDK publica o global `puter`.
  //
  // Este detalhe so aparece no Tampermonkey. Com "@grant GM_xmlhttpRequest" o
  // script roda num *isolated world*: o DOM, o document e o localStorage sao
  // compartilhados com a pagina, mas os GLOBALS que a pagina define nao sao. Um
  // <script> injetado executa no contexto da pagina, entao `window.puter` existe
  // la -- e daqui, dentro do sandbox, ele e undefined. O sintoma exato e "SDK
  // carregou sem definir puter" num instante em que a pagina tem o objeto na mao.
  //
  // `unsafeWindow` e a ponte que o gerenciador oferece justamente para alcancar
  // globals da pagina. Fora do gerenciador (harnesses, console) ele nao existe
  // e o `window.puter` normal basta.
  function puterGlobal() {
    try {
      if (typeof unsafeWindow !== 'undefined' && unsafeWindow && unsafeWindow.puter && unsafeWindow.puter.ai) {
        return unsafeWindow.puter;
      }
    } catch { /* politica do gerenciador pode bloquear o acesso */ }
    return (typeof window !== 'undefined' && window.puter && window.puter.ai) ? window.puter : null;
  }

  function puterPronto() {
    return !!puterGlobal();
  }

  function carregarPuter() {
    if (puterPronto()) return Promise.resolve(puterGlobal());
    if (puterPromise) return puterPromise;

    puterPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = PUTER_SDK;
      s.async = true;

      const inicio = Date.now();
      const limite = 20000;
      let encerrado = false;
      const finish = (fn) => {
        if (encerrado) return;
        encerrado = true;
        fn();
      };

      // Nao basta esperar o onload: o script pode ter executado e o SDK ainda
      // estar se inicializando (ele carrega outros pedacos). E o caso comum no
      // sandbox, onde o global so aparece depois. O que resolve e POLIR, com
      // prazo -- e nao falhar no primeiro instante em que o global nao existe.
      const sondar = () => {
        const p = puterGlobal();
        if (p) return finish(() => resolve(p));
        if (Date.now() - inicio > limite) {
          return finish(() => reject(new Error(
            `o script ${PUTER_SDK} carregou, mas o global "puter" nao apareceu em `
            + `${Math.round(limite / 1000)}s. Se o painel mostra "GM_xmlhttpRequest `
            + 'disponivel", o script esta no sandbox do gerenciador e o SDK precisa '
            + 'de "@grant unsafeWindow" no metadata -- confira a versao do script. '
            + 'Enquanto isso, volte o Backend para "HTTP direto".'
          )));
        }
        setTimeout(sondar, 150);
      };

      s.onload = sondar;
      s.onerror = () => finish(() => reject(new Error(`nao consegui carregar ${PUTER_SDK}`)));
      document.head.appendChild(s);

      // Se nem o onload chegar (CDN travada ou bloqueada por extensao),
      // sondar assim mesmo evita ficar mudo ate o timeout do agente.
      setTimeout(sondar, 300);
    }).catch((e) => {
      puterPromise = null; // permite tentar de novo no proximo ciclo
      throw e;
    });

    return puterPromise;
  }

  async function puterAssinou() {
    try {
      const p = await carregarPuter();
      return !!(await p.auth.isSignedIn());
    } catch {
      return false;
    }
  }

  // Entra no Puter. Precisa ser chamado por clique: o fluxo abre uma janela do
  // navegador, e um pop-up disparado pelo loop autonomo seria nao solicitado.
  async function puterEntrar() {
    const p = await carregarPuter();
    const r = await p.auth.signIn();
    const u = r?.user || (await p.auth.getUser().catch(() => null));
    return u?.username || u?.email || 'conta Puter';
  }

  // data URL -> File. E o caminho documentado para anexar imagem no Puter.
  function dataUrlParaFile(dataUrl, nome = 'grafico.jpg') {
    const [cabe, dados] = String(dataUrl).split(',');
    if (!dados) return null;
    const mime = (cabe.match(/^data:([^;]+)/) || [])[1] || 'application/octet-stream';
    const bin = atob(dados);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new File([bytes], mime.includes('png') ? 'grafico.png' : nome, { type: mime });
  }

  // Prompt de sistema + historico + enunciado num texto so (ver nota 2 do
  // bloco). Separado de askPuter para poder ser testado sem chamar a API.
  function puterPrompt(systemPrompt, userPrompt) {
    const partes = [];
    for (const m of [{ role: 'system', content: systemPrompt }, ...state.history.slice(-state.cfg.memoryTurns)]) {
      // 'system' PRECISA entrar no texto: com media so existe o overload de
      // prompt em string, entao o prompt de sistema some se for filtrado. E ele
      // que carrega o catalogo de acoes e as regras -- perder o sistema aqui
      // produz um modelo que devolve texto solto em vez do JSON do plano.
      if (m.role !== 'system' && m.role !== 'assistant' && m.role !== 'user') continue;
      const quem = m.role === 'system' ? 'Instrucoes' : m.role === 'assistant' ? 'Agente' : 'Usuario';
      partes.push(`${quem}: ${msgTexto(m)}`);
    }
    return partes.concat([userPrompt]).join('\n\n');
  }

  // Uma chamada ao Puter. Espelha o contrato de askAI: devolve o texto, ou
  // lanca.
  async function askPuter(systemPrompt, userPrompt, image) {
    const p = await carregarPuter();

    if (!(await p.auth.isSignedIn())) {
      const err = new Error(
        'Puter exige conta: clique em Config > "Entrar no Puter" uma vez. '
        + 'O modelo User-Pays usa o seu proprio plano, nao ha chave para colar.'
      );
      err.needPuterLogin = true;
      throw err;
    }

    const texto = puterPrompt(systemPrompt, userPrompt);

    const opcoes = {
      model: state.cfg.model,
      temperature: 0.2,
      // normalize:true garante message.content como string, em vez do formato
      // nativo do vendor (Anthropic devolve blocos). Assim extrairTexto() le
      // igual em qualquer modelo.
      normalize: true,
    };
    if (state.cfg.maxOutputTokens > 0) opcoes.max_tokens = state.cfg.maxOutputTokens;

    // (prompt, media, testMode, options) -- testMode false = chamada de verdade.
    const resp = image
      ? await p.ai.chat(texto, dataUrlParaFile(image.dataUrl), false, opcoes)
      : await p.ai.chat(texto, false, opcoes);

    const dados = { message: resp?.message || resp };
    state.narracao = extrairNarracao(dados);
    const conteudo = extrairTexto(dados);
    if (!conteudo) throw new Error('Puter devolveu resposta sem texto');
    return conteudo;
  }

  // O que o modelo DISSE, alem da ferramenta escolhida.
  //
  // Com tool calling nativo o `content` costuma vir null e a explicacao vai
  // para campos proprios do provider:
  //   reasoning_content -> DeepSeek, Groq, OpenRouter
  //   reasoning         -> OpenRouter
  //   content[] com partes "thinking" -> Cohere
  // Tudo isso era descartado. O log ficava "> select_choice ::" sem nada
  // depois, e nao havia como saber o que o modelo pretendia escrever.
  function extrairNarracao(data) {
    const msg = data?.choices?.[0]?.message || data?.message || data || {};
    const juntar = (v) => {
      if (typeof v === 'string') return v.trim();
      if (Array.isArray(v)) {
        return v
          .filter((p) => p && typeof p.text === 'string'
            && /^(text|thinking|reasoning|output_text)$/.test(p.type || ''))
          .map((p) => p.text)
          .join(' ')
          .trim();
      }
      return '';
    };

    const partes = [];
    for (const campo of ['reasoning_content', 'reasoning', 'thinking']) {
      const t = juntar(msg[campo]);
      if (t) partes.push(t);
    }
    const conteudo = juntar(msg.content) || juntar(data?.content);
    // O `content` entra por ULTIMO: quando o modelo escreve a explicacao e a
    // ferramenta juntas, e nele que a frase aparece.
    if (conteudo) partes.push(conteudo);
    return partes.join(' ').replace(/\s+/g, ' ').trim();
  }

  function extrairTexto(data) {
    // Tool calling nativo vem ANTES de qualquer texto.
    //
    // Quando o modelo responde com tool_calls, `content` costuma vir null
    // (OpenAI) ou conter so o raciocinio (alguns providers). Normalizamos a
    // chamada para o MESMO JSON que o modo texto produz, e daqui para baixo
    // nada mais precisa saber a diferenca: decide()/parseJsonLoose() leem igual.
    //
    // Vantagem de normalizar em vez de tratar tool call como um caminho
    // paralelo: os dois formatos passam pelo mesmo validador, pela mesma
    // conferencia de choice_text e pela mesma memoria. Nao ha um "modo tools"
    // com regras proprias para divergir.
    //
    // Varias tool_calls na MESMA resposta viram {"actions":[...]} -- que e
    // exatamente a semantica de plano condicional: executar em ordem, parando
    // no primeiro passo que nao se aplicar mais.
    const chamadas = data?.choices?.[0]?.message?.tool_calls;
    if (Array.isArray(chamadas) && chamadas.length) {
      const passos = chamadas
        .filter((t) => t?.function?.name)
        .map((t) => {
          let args = {};
          try { args = JSON.parse(t.function.arguments || '{}'); } catch { args = {}; }
          return { action: t.function.name, params: args };
        });
      if (passos.length) {
        const pensamento = extrairNarracao(data);
        const corpo = passos.length === 1
          ? { action: passos[0].action, params: passos[0].params, thought: pensamento || '' }
          : { actions: passos, thought: pensamento || '' };
        return JSON.stringify(corpo);
      }
    }
    return extrairTextoSimples(data);
  }

  function extrairTextoSimples(data) {
    const juntar = (partes) => (Array.isArray(partes)
      ? partes
        .filter((p) => p && p.type === 'text' && typeof p.text === 'string')
        .map((p) => p.text)
        .join('')
        .trim()
      : '');

    const conteudo = data?.choices?.[0]?.message?.content;
    if (typeof conteudo === 'string' && conteudo.trim()) return conteudo.trim();

    // Ordem: os formatos de lista primeiro (o `content` string ja foi tratado
    // acima), depois os formatos de string fora de `choices`.
    //
    // Duas formas que este fallback NAO cobria e que o teste pegou:
    //   - Cohere chamado pelo endpoint OpenAI-compat devolve
    //     choices[0].message.content como LISTA, e nao como data.message;
    //   - o Puter devolve { message: { content: "texto" } } -- string fora de
    //     choices, que caia em juntar() e voltava vazia.
    // Nos dois casos o resultado era '' e o chamador dizia "resposta sem texto"
    // com a API respondendo 200.
    return juntar(conteudo)                 // choices[0].message.content em partes
      || juntar(data?.message?.content)     // Cohere nativo: message.content[]
      || juntar(data?.content)              // Anthropic nativo: content[]
      || (typeof data?.message?.content === 'string' ? data.message.content.trim() : '') // Puter
      || (typeof data?.content === 'string' ? data.content.trim() : '');
  }

  // Parte de imagem da mensagem. Existem dois formatos no mundo e nenhum
  // provider documenta qual aceita:
  //
  //   'objeto'  {type:'image_url', image_url:{url:'...'}}   OpenAI, Groq,
  //              Together, ollama, LM Studio.
  //   'string'  {type:'image_url', image_url:'...'}          NVIDIA NIM
  //              (ai.api.nvidia.com/v1/vlm/...) -- que aceita imagem/svg+xml
  //              e exige EXATAMENTE uma imagem por request.
  //
  // Comecamos no formato objeto e, se a API recusar o schema, marcamos o
  // outro e tentamos de novo. Descobrir por tentativa e mais barato que
  // adivinhar pelo hostname: o mesmo modelo muda de formato entre versoes.
  function imagePart(dataUrl, estilo) {
    return estilo === 'string'
      ? { type: 'image_url', image_url: dataUrl }
      : { type: 'image_url', image_url: { url: dataUrl } };
  }

  // Texto de uma mensagem, qualquer que seja o formato do content.
  function msgTexto(m) {
    if (typeof m.content === 'string') return m.content;
    if (Array.isArray(m.content)) {
      return m.content.filter((p) => p && p.type === 'text').map((p) => p.text).join('\n');
    }
    return '';
  }

  // Alguns endpoints (NVIDIA NIM /v1/vlm/...) aceitam no maximo UMA mensagem
  // por request: o prompt de sistema tem de ser dobrado dentro da unica
  // mensagem de usuario, junto com o historico.
  // Partes de imagem, com o rotulo de cada uma ANTES dela.
  //
  // O rotulo e o que permite a IA ligar cada figura ao seu indice ("IMAGEM DA
  // ALTERNATIVA [2]"). Sem ele, varias imagens chegam como um bloco anonimo e
  // a IA nao sabe qual e qual -- exatamente o problema que o texto vazio das
  // alternativas-imagem ja causava.
  function partesDeImagem(imagens) {
    const partes = [];
    for (const im of imagens || []) {
      if (!im || !im.dataUrl) continue;
      if (im.rotulo) partes.push({ type: 'text', text: im.rotulo });
      partes.push(imagePart(im.dataUrl, state.imgStyle));
    }
    return partes;
  }

  function buildMessages(systemPrompt, userPrompt, imagens) {
    const partes = partesDeImagem(imagens);

    if (state.singleMessage) {
      const hist = state.history.slice(-state.cfg.memoryTurns);
      let texto = systemPrompt;
      if (hist.length) {
        texto += '\n\nCONVERSA ANTERIOR (o que ja foi feito):\n'
          + hist
            .map((m) => `${m.role === 'assistant' ? 'Agente' : 'Usuario'}: ${msgTexto(m)}`)
            .join('\n')
          + '\n\n---\n\n';
      }
      texto += userPrompt;
      return [{
        role: 'user',
        content: partes.length ? [{ type: 'text', text: texto }, ...partes] : texto,
      }];
    }

    const ultima = partes.length
      ? { role: 'user', content: [{ type: 'text', text: userPrompt }, ...partes] }
      : { role: 'user', content: userPrompt };

    return [
      { role: 'system', content: systemPrompt },
      ...state.history.slice(-state.cfg.memoryTurns),
      ultima,
    ];
  }

  // POST JSON que sobrevive a provider sem CORS.
  //
  // A pagina do Khan e uma origem como qualquer outra, e o browser so entrega
  // a resposta se o provider mandar Access-Control-Allow-Origin. O NVIDIA NIM,
  // por exemplo, nao manda: nem um OPTIONS passa, e o fetch morre com
  // "Failed to fetch" sem status nenhum -- indistinguivel de rede fora do ar.
  //
  // O Userscript tem a saida: GM_xmlhttpRequest, que sai da extension e por
  // isso ignora CORS. A estrategia aqui e FALLBACK, nao preferencia:
  //   - primeiro fetch (funciona em Groq/OpenAI/Pollinations, ja testado);
  //   - se o fetch falhar como rede E o GM existir, repete pelo GM e marca a
  //     sessao para ir direto nele nas proximas chamadas.
  // Quando o script roda fora de um gerenciador (harnesses, colar no console),
  // nao ha GM: ai o fetch e tudo que temos, e o erro sai com a dica certa.
  function gmAvailable() {
    return typeof GM_xmlhttpRequest === 'function';
  }

  function gmPostJson(url, headers, body, signal) {
    return new Promise((resolve, reject) => {
      let req = null;
      let settled = false;
      const finish = (fn, arg) => {
        if (settled) return;
        settled = true;
        fn(arg);
      };

      try {
        req = GM_xmlhttpRequest({
          method: 'POST',
          url,
          headers,
          data: body,
          timeout: state.cfg.requestTimeoutMs,
          onload: (r) => finish(resolve, {
            ok: r.status >= 200 && r.status < 300,
            status: r.status,
            text: async () => r.responseText || '',
            json: async () => JSON.parse(r.responseText || '{}'),
          }),
          onerror: () => finish(reject, new Error('GM_xmlhttpRequest: erro de rede')),
          ontimeout: () => finish(reject, new Error(
            `GM_xmlhttpRequest: sem resposta em ${Math.round(state.cfg.requestTimeoutMs / 1000)}s `
            + '(o endpoint medido responde em 2 a 4 s, entao isso nao e lentidao do modelo). '
            + 'Diagnostico: (1) clique em Config > Testar -- se tambem travar, o problema '
            + 'e o caminho GM/rede do navegador, nao o agente; (2) se so o agente travar, '
            + 'desmarque "Enviar o grafico/imagem para a IA": o corpo cai de ~32 KB para '
            + '~3 KB e sobra REQUEST grande; (3) VPN/proxy corporativo bloqueia '
            + 'integrate.api.nvidia.com -- nesse caso use um provider com CORS, como '
            + 'Groq, OpenAI, OpenRouter ou Cohere.'
          )),
          onabort: () => finish(reject, new Error('GM_xmlhttpRequest: cancelado')),
        });
      } catch (e) {
        finish(reject, e);
        return;
      }

      // O AbortController do fetch tambem precisa mandar no GM.
      if (signal) {
        if (signal.aborted) { try { req?.abort(); } catch {} }
        signal.addEventListener('abort', () => { try { req?.abort(); } catch {} });
      }
    });
  }

  async function postJson(url, headers, body, signal) {
    // Ja sabemos que o fetch e barrado nesta sessao: vai direto pelo GM.
    if (state.useGm && gmAvailable()) return gmPostJson(url, headers, body, signal);

    try {
      return await fetch(url, { method: 'POST', headers, body, signal });
    } catch (e) {
      // "Failed to fetch" e o sintoma de CORS; "aborted" e timeout e nao
      // falta de permissao. So o primeiro caso vira tentativa pelo GM.
      const cors = /failed to fetch|networkerror|load failed/i.test(e.message || '');
      if (cors && !signal?.aborted) {
        if (gmAvailable()) {
          state.useGm = true;
          log('O provider barra CORS; seguindo via GM_xmlhttpRequest.', 'warn');
          return gmPostJson(url, headers, body, signal);
        }
        // Sem GM nao ha como contornar: o navegador esconde a resposta. As
        // duas causas reais sao "@grant none" no metadata e script nao
        // recarregado depois da edicao -- as duas produzem este erro exato,
        // e nenhuma delas e problema do endpoint.
        throw new Error(
          `${e.message || 'Falha de rede'} — este endpoint nao libera CORS, entao o `
          + 'navegador bloqueia a resposta. Provider assim exige o GM_xmlhttpRequest, '
          + 'que so existe no gerenciador de userscript: confira se o metadata tem '
          + '"@grant GM_xmlhttpRequest" SEM "@grant none", salve o script e RECARREGUE '
          + 'a pagina do Khan (o Tampermonkey so injeta a API na proxima carga da aba). '
          + 'Se persistir, use um provider com CORS: Groq, OpenAI, OpenRouter ou Cohere.'
        );
      }
      throw e;
    }
  }

  // GET JSON com o mesmo recuo de CORS do POST. Precisa existir separado
  // porque `gmPostJson` e POST-only: o catalogo de modelos e um GET, e sem
  // esta versao providers sem CORS (NVIDIA) nao teriam lista nenhuma.
  function gmGetJson(url, headers) {
    return new Promise((resolve, reject) => {
      let req = null;
      let settled = false;
      const finish = (fn, arg) => {
        if (settled) return;
        settled = true;
        fn(arg);
      };
      try {
        req = GM_xmlhttpRequest({
          method: 'GET',
          url,
          headers,
          timeout: state.cfg.requestTimeoutMs,
          onload: (r) => finish(resolve, {
            ok: r.status >= 200 && r.status < 300,
            status: r.status,
            text: async () => r.responseText || '',
            json: async () => JSON.parse(r.responseText || '{}'),
          }),
          onerror: () => finish(reject, new Error('GM_xmlhttpRequest: erro de rede')),
          ontimeout: () => finish(reject, new Error('GM_xmlhttpRequest: sem resposta')),
          onabort: () => finish(reject, new Error('GM_xmlhttpRequest: cancelado')),
        });
      } catch (e) {
        finish(reject, e);
        return;
      }
      void req;
    });
  }

  async function getJson(url, headers) {
    if (state.useGm && gmAvailable()) return gmGetJson(url, headers);
    try {
      return await fetch(url, { method: 'GET', headers });
    } catch (e) {
      const cors = /failed to fetch|networkerror|load failed/i.test(e.message || '');
      if (cors && gmAvailable()) {
        state.useGm = true;
        log('O provider barra CORS; seguindo via GM_xmlhttpRequest.', 'warn');
        return gmGetJson(url, headers);
      }
      throw e;
    }
  }

  // O que fazer quando o provider responde 402 (Payment Required).
  //
  // Nao e erro de formato nem de rede: a conta esta sem saldo, ou a cota do
  // plano gratuito acabou. Repetir nao resolve, entao o agente para e diz o que
  // fazer -- antes disso o usuario via so "Falha na IA: HTTP 402", que nao
  // ajuda em nada.
  function dicaDePagamento(endpoint, model) {
    const ep = String(endpoint || '');
    const m = model ? ` (modelo "${model}")` : '';
    if (/pollinations/i.test(ep)) {
      return `Sem creditos no Pollinations${m}. O modo anonimo foi descontinuado: gere uma key `
        + 'em enter.pollinations.ai/keys, cole em Config e confira o saldo da conta. '
        + 'Alternativas gratuitas: Groq, OpenRouter (modelos ":free"), Cohere ou Gemini.';
    }
    if (/openrouter/i.test(ep)) {
      return `Creditos insuficientes no OpenRouter${m}. Modelos com sufixo ":free" tem cota `
        + 'diaria propria, separada do saldo -- se a cota acabou, espere o reset ou escolha '
        + 'outro ":free". Modelo pago exige saldo na conta.';
    }
    if (/api\.openai\.com/i.test(ep)) {
      return `A conta OpenAI esta sem creditos${m}. Adicione credito em platform.openai.com, ou `
        + 'use um provider gratuito: Groq, OpenRouter, Cohere ou Gemini.';
    }
    return `O provider recusou por falta de pagamento ou creditos${m} (HTTP 402). Confira o saldo `
      + 'da conta e se a chave pertence a ela. Se o plano gratuito acabou, use Groq, '
      + 'OpenRouter (":free"), Cohere ou Gemini.';
  }

  async function askAI(systemPrompt, userPrompt, image, extras = []) {
    const imagens = [image, ...(extras || [])].filter((i) => i && i.dataUrl);

    // Backend Puter: nada de endpoint nem chave. O SDK roda no navegador, entao
    // nao ha CORS a negociar e nao ha Bearer para mandar.
    //
    // O `media` do Puter aceita UMA imagem por chamada, entao alternativas com
    // imagem nao passam por ele -- o aviso diz isso em vez de mandar so metade
    // em silencio.
    if (state.cfg.backend === 'puter') {
      if (imagens.length > 1) {
        log(`O Puter aceita uma imagem por chamada; enviando so a principal (${imagens.length - 1} alternativa(s) com imagem ficaram de fora).`, 'warn');
      }
      return askPuter(systemPrompt, userPrompt, imagens[0] || null);
    }

    const messages = buildMessages(systemPrompt, userPrompt, imagens);
    // Tool calling nativo. Onde o provider aceita, a IA escolhe a ferramenta por
    // schema em vez de escrever JSON dentro do texto -- e erra menos o formato.
    // Onde nao aceita, o corpo sai sem `tools` e o JSON no texto continua
    // valendo (ver a deteccao de toolsUnsupported mais abaixo).
    const usaTools = state.cfg.toolCalling && !state.toolsDead;
    const body = JSON.stringify({
      model: state.cfg.model,
      messages,
      temperature: 0.2,
      // teto de tokens de SAIDA. Sem isso o provider usa o padrao dele, que em
      // modelo grande com raciocinio significa gerar milhares de tokens antes
      // do JSON -- foi assim que uma chamada de 32 KB passou de 60 s e o agente
      // pareceu travar. O plano e um objeto JSON de poucas linhas; 2000 tokens
      // dao folga inclusive para o bloco de raciocinio dos modelos que o
      // devolvem (Cohere, por exemplo, manda uma parte "thinking").
      //
      // maxOutputTokens <= 0 omite o campo de proposito: quem usa provider que
      // tem o proprio controle de tamanho precisa sair do nosso.
      ...(state.cfg.maxOutputTokens > 0 ? { max_tokens: state.cfg.maxOutputTokens } : {}),
      ...(usaTools ? { tools: toolSchemas() } : {}),
      ...(usaTools && !state.toolChoiceDead ? { tool_choice: 'auto' } : {}),
    });

    // O fallback legado (text.pollinations.ai) so faz sentido no Pollinations.
    // Com um provider de terceiro -- Groq, OpenAI, Together, ollama -- tentar
    // o host do Pollinations depois de falhar produz exatamente o sintoma de
    // "troquei o modelo e ele continua usando o antigo": a chamada vai para o
    // host errado e o erro do provider nunca aparece. Aqui so aceitamos o
    // fallback quando o endpoint configurado pertence ao Pollinations.
    const endpoint = String(state.cfg.endpoint || '');
    const isPollinations = /(^|\/\/|\.)(gen|text)\.pollinations\.ai(\/|$|:)/i.test(endpoint);
    const candidates = [endpoint];
    if (isPollinations) candidates.push(state.cfg.fallbackEndpoint);
    const endpoints = candidates.filter((u, i, arr) => u && arr.indexOf(u) === i);

    let lastError = null;
    let authError = null;
    let visionError = null;
    let styleError = null;
    let singleMsgError = null;
    let toolError = null;
    let extrasError = null;
    let paymentError = null;
    let transientError = null;

    for (const url of endpoints) {
      for (let attempt = 1; attempt <= state.cfg.maxRetries; attempt++) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), state.cfg.requestTimeoutMs);
        const t0 = Date.now();

        // Uma chamada com imagem de 32 KB para um modelo de 31 B leva dezenas de
        // segundos, e o painel ficava mudo esse tempo inteiro -- parecia trava.
        // Este aviso so aparece se a resposta nao vier em 4 s, entao no
        // caminho rapido nao polui o log.
        const slow = setTimeout(() => {
          log(
            `A IA ainda nao respondeu (${Math.round((Date.now() - t0) / 1000)}s). `
            + `${Math.round(body.length / 1024)} KB, modelo ${state.cfg.model}`
            + `${imagens.length ? `, ${imagens.length} imagem(ns)` : ''}.`,
            'warn'
          );
        }, 4000);

        try {
          const headers = { 'Content-Type': 'application/json' };
          if (state.cfg.apiKey) headers.Authorization = `Bearer ${state.cfg.apiKey}`;

          const res = await postJson(url, headers, body, controller.signal);

          if (!res.ok) {
            const detail = await res.text().catch(() => '');
            const err = new Error(`HTTP ${res.status} :: ${detail.slice(0, 140)}`);
            err.status = res.status;
            // Guarda o corpo inteiro: e dele que sai a distincao entre "cota
            // gratuita estourou" e "demanda passageira", que pedem esperas
            // diferentes.
            err.detail = detail.slice(0, 400);
            // 401/403 = credencial. Nao adianta repetir nem trocar de host.
            if (res.status === 401 || res.status === 403) {
              err.auth = true;
              if (!state.cfg.apiKey) {
                err.hint = 'Sem API key. Pegue uma em enter.pollinations.ai e clique em Config no painel.';
              } else {
                err.hint = 'API key recusada. Confira a key em Config no painel.';
              }
              authError = err;
            }
            // 402 = Payment Required. Nao e formato nem rede: e a CONTA sem
            // saldo/cota. Repetir nao resolve e insistir so gasta tempo -- por
            // isso ele para com uma explicacao, em vez de subir como
            // "Falha na IA: HTTP 402", que nao diz o que fazer.
            if (res.status === 402) {
              err.payment = true;
              err.hint = dicaDePagamento(state.cfg.endpoint, state.cfg.model);
              paymentError = err;
            }
            // 400 numa mensagem que LEVAVA imagem quase sempre e "este modelo
            // nao e multimodal". Nao adianta repetir o mesmo corpo, e o
            // chamador usa esta marca para tentar de novo sem a imagem.
            if (imagens.length && res.status === 400
                && /(image|vision|multimodal|content must be a string|image_url)/i.test(detail)) {
              err.visionUnsupported = true;
              visionError = err;
            }
            // 502/503/504 no free tier e demanda, nao erro nosso: o Gemini
            // responde "This model is currently experiencing high demand" e o
            // OpenRouter free faz o mesmo. Tentar outro host nao adianta, mas
            // insistir no MESMO host depois de alguns segundos resolve.
            if (res.status === 502 || res.status === 503 || res.status === 504) {
              err.transient = true;
              transientError = err;
            }
            // Schema da parte de imagem recusado: o provider quer
            // image_url como STRING, nao como objeto. Nao e falta de visao --
            // e formato. O chamador troca o estilo e repete a chamada.
            if (imagens.length && (res.status === 400 || res.status === 422)
                && /(string_type|UserImageContent|Extra inputs are not permitted|Input should be a valid string)/i.test(detail)
                && state.imgStyle !== 'string') {
              err.imgStyleWrong = true;
              styleError = err;
            }
            // Varias imagens recusadas. Nao e "sem visao" (a imagem principal
            // passa): e LIMITE DE QUANTIDADE. Descartamos as alternativas e
            // seguimos com a figura principal, em vez de perder a visao inteira
            // por causa de uma restricao que so vale para as extras.
            if (imagens.length > 1 && (res.status === 400 || res.status === 422)
                && /((exactly|at most|only|maximum of|max)\s*(one|1)\s*image|too many images|images?[^.]{0,20}not supported)/i.test(detail)
                && !state.extrasDead) {
              err.tooManyImages = true;
              extrasError = err;
            }
            // Endpoint que aceita UMA mensagem so: o prompt de sistema tem de
            // entrar dentro da mensagem de usuario. A tentativa e uma so,
            // porque mudar o corpo e o caminho, nao insistir no mesmo erro.
            if ((res.status === 400 || res.status === 422)
                && /at most 1 item|too_long|max_length=1|should have at most/i.test(detail)
                && !state.singleMessage) {
              err.singleMessageWrong = true;
              singleMsgError = err;
            }
            // Tool calling recusado. Descemos UM degrau por vez, porque os
            // providers recusam partes diferentes: alguns aceitam `tools` mas
            // nao conhecem `tool_choice`, outros nao conhecem nenhum dos dois.
            // Tirar tudo de uma vez perderia o caso intermediario, que e o mais
            // comum entre os compativeis parciais -- e que ainda ganha o
            // beneficio do tool calling.
            if (usaTools && (res.status === 400 || res.status === 422)) {
              const falaDeChoice = /tool_choice/i.test(detail);
              const falaDeTools = /\btools?\b|tool_calls|function calling|tool use|tool use is not supported/i.test(detail);
              if (falaDeChoice && !state.toolChoiceDead) {
                err.toolChoiceWrong = true;
                toolError = err;
              } else if (falaDeTools && !state.toolsDead) {
                err.toolsUnsupported = true;
                toolError = err;
              }
            }
            throw err;
          }

          const data = await res.json();
          // Guarda o que o modelo DISSE antes de extrair o plano: com tool
          // calling o texto vem em campos proprios e seria perdido.
          state.narracao = extrairNarracao(data);
          const content = extrairTexto(data);
          if (content) return content;
          throw new Error(
            `resposta sem texto reconhecivel (chaves: ${Object.keys(data || {}).join(',') || 'corpo vazio'})`
          );
        } catch (err) {
          lastError = err;
          if (err.auth) break;
          // Sem saldo (402): o mesmo corpo falha igual, e insistir so gasta tempo.
          if (err.payment) break;
          // Modelo sem visao: repetir o mesmo corpo nao muda nada, e o erro
          // de 400 se repete em cada tentativa.
          if (err.visionUnsupported) break;
          // Formato de imagem errado: o mesmo corpo falha igual sempre.
          if (err.imgStyleWrong) break;
          // Endpoint de uma mensagem so: insistir no corpo de 2 mensagens
          // devolve sempre o mesmo 422.
          if (err.singleMessageWrong) break;
          // Tool calling: o corpo e que esta errado, entao insistir devolve o
          // mesmo 400. O chamador troca o corpo e repete.
          if (err.toolChoiceWrong || err.toolsUnsupported) break;
          // Excesso de imagens: o chamador tira as alternativas e repete.
          if (err.tooManyImages) break;

          // 429 e limite de cota, nao erro da requisicao: insistir em 800 ms
          // so gasta as tentativas restantes. O plano free do Groq precisa de
          // dezenas de segundos, entao a espera cresce de verdade -- e a
          // retentativa depois de "Incorrect", que era justamente o momento em
          // que mais importa, parava no 429 antes de acontecer.
          //
          // 502/503/504 entram no mesmo balde, com espera menor: e demanda do
          // free tier ("high demand"), nao cota estourada, e costuma passar em
          // poucos segundos.
          if (err.status === 429 || err.transient) {
            // Cota de plano gratuito (o free tier do Gemini, por exemplo) e
            // medida em POUCAS chamadas por MINUTO: esperar 6s nao resolve, o
            // balde so volta a encher perto de 60s. Sem distinguir isso, o
            // usuario le "limite de cota" e acha que o problema e a chave.
            const cotaGratuita = /free_tier|free tier|quota exceeded/i.test(err.detail || '');
            const teto = err.transient ? 15000 : (cotaGratuita ? 60000 : 45000);
            const base = err.transient ? 3000 : (cotaGratuita ? 20000 : 6000);
            const espera = Math.min(teto, base * 2 ** (attempt - 1));
            log(
              `${err.transient ? 'Indisponivel no momento' : 'Limite de cota'}`
              + `${cotaGratuita ? ' do plano gratuito' : ''} (${err.status}). Esperando `
              + `${Math.round(espera / 1000)}s antes da tentativa ${attempt + 1}/${state.cfg.maxRetries}.`,
              'warn'
            );
            if (cotaGratuita && attempt === 1) {
              log(
                'A cota gratuita deste provider e medida em poucas chamadas por minuto. '
                + 'O agente faz cerca de uma chamada por acao, entao ela acaba rapido: '
                + 'vale esperar o minuto virar ou usar um provider com cota maior.',
                'warn'
              );
            }
            if (attempt < state.cfg.maxRetries) {
              await sleep(espera);
              continue;
            }
            break;
          }

          log(
            `IA falhou (${err.message}) apos ${Math.round((Date.now() - t0) / 1000)}s `
            + `- tentativa ${attempt}/${state.cfg.maxRetries}`,
            'warn'
          );
          await sleep(800 * attempt);
        } finally {
          clearTimeout(timer);
          clearTimeout(slow);
        }
      }
      if (authError || visionError || styleError || singleMsgError || toolError || extrasError || paymentError) break;
    }
    // transientError nao entra no break acima de proposito: 502/503/504 sao
    // demanda passageira do free tier, entao o laco tenta de novo. So chega
    // aqui se TODAS as tentativas pegaram indisponibilidade -- e ai a mensagem
    // precisa dizer isso, em vez de deixar o usuario achando que errou a key.
    if (transientError && !lastError) lastError = transientError;
    if (transientError && lastError === transientError) {
      throw new Error(
        `O provider esta indisponivel no momento (${transientError.status}): `
        + 'demanda alta no free tier, nao erro de configuracao. Tente de novo em '
        + 'alguns minutos ou troque de modelo/provider.'
      );
    }
    throw authError || paymentError || visionError || styleError || singleMsgError || toolError || extrasError
      || new Error(`Falha na IA: ${lastError?.message || 'desconhecida'}`);
  }

  /* ===========================================================================
   * CATALOGO DE FERRAMENTAS
   * ---------------------------------------------------------------------------
   * Fonte unica. Daqui saem TRES coisas que antes viviam escritas em separado e
   * podiam divergir em silencio:
   *   1. o campo `tools` do request (tool calling nativo, onde o provider aceita);
   *   2. a lista de acoes do prompt de texto (o caminho de quem NAO aceita tools);
   *   3. a lista "ACOES DISPONIVEIS NESTE MOMENTO" daquele snapshot.
   *
   * O bug que isto elimina: uma acao nova adicionada em ACTIONS e esquecida no
   * texto do prompt simplesmente nao existia para a IA -- e esquecida em `avail`
   * aparecia como indisponivel. Nenhum dos dois casos dava erro; so silencio.
   *
   * ACTIONS (a implementacao) fica separado de proposito: este catalogo DESCREVE,
   * ACTIONS EXECUTA. Um teste garante que os dois conjuntos batem.
   *
   * `available(snap)` e a pre-condicao. Ela tambem aborta planos condicionais: se
   * o passo seguinte nao se aplica mais a pagina, o plano para ali em vez de
   * clicar em algo que nao existe.
   * ======================================================================== */
  const TOOLS = [
    {
      name: 'select_choice',
      kind: 'answer',
      summary: '{"choice_index":N,"choice_text":"<texto EXATO da alternativa N>"}'
        + ' -> marca a alternativa de indice N (0-based). Copie choice_text da'
        + ' lista ALTERNATIVAS, sem reescrever.',
      params: {
        type: 'object',
        properties: {
          choice_index: { type: 'integer', description: 'Indice 0-based da alternativa, na ordem em que aparece na tela.' },
          choice_text: { type: 'string', description: 'Texto exato da alternativa escolhida, copiado da lista ALTERNATIVAS.' },
        },
        required: ['choice_index', 'choice_text'],
      },
      available: (s) => !!s.choices.length,
    },
    {
      name: 'type_answer',
      kind: 'answer',
      summary: '{"inputs":[{"index":N,"text":"valor"}]} -> preenche campos de TEXTO.',
      params: {
        type: 'object',
        properties: {
          inputs: {
            type: 'array',
            description: 'Um item por campo de texto a preencher.',
            items: {
              type: 'object',
              properties: {
                index: { type: 'integer', description: 'Indice do campo na lista CAMPOS DE RESPOSTA.' },
                text: { type: 'string', description: 'Valor a digitar.' },
              },
              required: ['index', 'text'],
            },
          },
        },
        required: ['inputs'],
      },
      available: (s) => s.inputs.some((f) => f.kind === 'text'),
    },
    {
      name: 'build_expression',
      kind: 'answer',
      summary: '{"latex":"\\\\frac{1}{2}"} -> escreve em campo de MATEMATICA (MathQuill).',
      params: {
        type: 'object',
        properties: {
          latex: {
            type: 'string',
            description: 'Expressao em LaTeX. O teclado do Khan aceita: 0-9, ponto,'
              + ' + - x, parenteses, fracao, potencia, raiz, porcentagem, negativo e a letra x.',
          },
        },
        required: ['latex'],
      },
      available: (s) => s.inputs.some((f) => f.kind === 'math' || f.kind === 'text'),
    },
    {
      name: 'calculate',
      kind: 'util',
      summary: '{"expression":"9/m+4","vars":{"m":3}} -> calcula a aritmetica e'
        + ' devolve o numero. Use SEMPRE para contas: de cabeca voce erra, aqui nao.',
      params: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'Expressao aritmetica: + - * / ^ %, parenteses, sqrt, abs,'
              + ' round, floor, ceil, min, max, log, ln, exp, pi, e.',
          },
          vars: {
            type: 'object',
            description: 'Opcional: valor de cada letra da expressao, ex. {"m":3}.',
            additionalProperties: { type: 'number' },
          },
        },
        required: ['expression'],
      },
      available: () => true,
    },
    {
      name: 'click_check',
      kind: 'nav',
      summary: '{} -> clica em "Verificar" (use depois de responder).',
      params: { type: 'object', properties: {}, required: [] },
      available: (s) => !!s.buttons.check,
    },
    {
      name: 'click_next',
      kind: 'nav',
      summary: '{} -> clica em "Continuar" para avancar.',
      params: { type: 'object', properties: {}, required: [] },
      available: (s) => !!s.buttons.next,
    },
    {
      name: 'skip',
      kind: 'nav',
      summary: '{} -> clica em "Pular". Pula uma ETAPA de uma questao em varias'
        + ' partes; de novo, passa de tela.',
      params: { type: 'object', properties: {}, required: [] },
      available: (s) => !!s.buttons.skip,
    },
    {
      name: 'start_practice',
      kind: 'nav',
      summary: '{} -> so quando a pagina mostrar a tela de inicio da unidade.',
      params: { type: 'object', properties: {}, required: [] },
      available: (s) => !!findStartButton() && !s.choices.length,
    },
    {
      name: 'hint',
      kind: 'util',
      summary: '{} -> pede uma dica. O TEXTO da dica volta para voce na proxima'
        + ' observacao: leia e use, em vez de pedir de novo.',
      params: { type: 'object', properties: {}, required: [] },
      available: (s) => !!s.buttons.hint,
    },
    {
      name: 'ask_user',
      kind: 'meta',
      summary: '{"question":"..."} -> a pergunta e ambigua demais (grafico/imagem sem texto).',
      params: {
        type: 'object',
        properties: { question: { type: 'string', description: 'A pergunta, em portugues, curta.' } },
        required: ['question'],
      },
      available: () => true,
    },
    {
      name: 'wait',
      kind: 'meta',
      summary: '{} -> nao ha acao clara; so reobserve a pagina.',
      params: { type: 'object', properties: {}, required: [] },
      available: () => true,
    },
  ];

  // A lista de acoes do prompt, gerada do catalogo. A coluna alinhada e o
  // formato em que a lista sempre foi escrita a mao; padEnd preserva isso.
  function promptDeAcoes() {
    return TOOLS.map((t) => `- ${t.name.padEnd(17)} ${t.summary}`);
  }

  // O catalogo no formato que a API de tool calling espera.
  //
  // Toda ferramenta ganha um `thought` OPCIONAL. E o que faz o modelo FALAR e
  // AGIR na mesma chamada: sem ele, o tool calling devolve so a chamada, o
  // `content` vem null, e o log fica "> select_choice ::" sem nada depois --
  // nao da para saber o que o modelo pretendia escrever.
  //
  // E opcional de proposito: exigir um campo extra em toda chamada aumenta a
  // chance de o modelo errar o formato, e a explicacao e util, nao obrigatoria.
  function toolSchemas() {
    return TOOLS.map((t) => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.summary,
        parameters: {
          ...t.params,
          properties: {
            ...(t.params.properties || {}),
            thought: {
              type: 'string',
              description: 'Uma frase curta em portugues dizendo o que voce vai fazer e por que, '
                + 'ANTES de agir. Aparece no log do usuario ao lado da acao, e e o que permite '
                + 'conferir o que voce pretendeu escrever.',
            },
          },
        },
      },
    }));
  }

  // Quais acoes fazem sentido NESTE snapshot. `available` pode lancar (uma
  // consulta ao DOM pode falhar no meio de um re-render do React), e uma
  // excecao aqui nao pode derrubar a montagem do prompt: vira indisponivel.
  function acoesDisponiveis(snap) {
    return TOOLS.filter((t) => {
      try { return !!t.available(snap); } catch { return false; }
    }).map((t) => t.name);
  }

  function achaTool(nome) {
    return TOOLS.find((t) => t.name === nome) || null;
  }

  /* ===========================================================================
   * PROMPT DO AGENTE
   * ======================================================================== */
  const SYSTEM_PROMPT = [
    'Voce e o cerebro de um agente que opera o Khan Academy por tras de um usuario humano.',
    'Voce recebe o estado atual da pagina e devolve UM UNICO objeto JSON puro.',
    'Sem markdown, sem cercas de codigo, sem texto antes ou depois do JSON.',
    '',
    'Formato obrigatorio:',
    '{"action":"<acao>","params":{...},"thought":"<curto, em portugues>"}',
    '',
    'Acoes e como preencher "params":',
    ...promptDeAcoes(),
    '',
    'VARIAS ACOES NUMA CHAMADA (economiza chamadas):',
    'Devolva {"actions":[...]} para encadear o ciclo inteiro numa chamada so. O',
    'script executa em ordem e PARA no primeiro passo que nao se aplicar mais a',
    'pagina -- se a resposta errou, nao existe "Continuar" e o click_next e',
    'descartado sozinho. Exemplo:',
    '  {"actions":[{"action":"select_choice","params":{"choice_index":2,"choice_text":"..."}},',
    '              {"action":"click_check"},{"action":"click_next"}],',
    '   "thought":"..."}',
    'Se voce mandar SO a resposta (sem click_check), o script clica em Verificar',
    'por conta propria -- esse e o caminho seguro quando ainda tem duvida.',
    '',
    'Como o Khan funciona de verdade (verificado no site):',
    '- Campo marcado "MATEMATICA" (math) NAO e uma caixinha de texto: e um editor',
    '  de formula. Ele aceita QUALQUER resposta, inclusive um numero puro -- mande',
    '  sempre "build_expression" com o valor em LaTeX ({"latex":"18"} para dezoito,',
    '  {"latex":"\\\\frac{1}{2}"} para um meio). "type_answer" nao funciona nele.',
    '- O teclado matematico do Khan so oferece estes digitos e operadores:',
    '  0-9, ponto, mais, menos, vezes, parenteses, fracao, potencia, raiz,',
    '  porcentagem, negativo - e a LETRA "x" (nenhuma outra letra existe).',
    '  Se a resposta precisar de outra letra, prefira "ask_user".',
    '- Depois de "click_check" o botao vira "Tentar novamente" enquanto errar.',
    '  Quando a prova esta CORRETA o Khan avanca sozinho ou mostra "Continuar".',
    '  Nunca clique em "click_check" duas vezes seguidas sem responder de novo.',
    '',
    'Regras:',
    '- Nunca use indice fora da faixa informada na pagina.',
    '- Explique SEMPRE, em UMA linha curta em portugues, o que voce vai fazer e por',
    '  que. Quando a resposta vier por ferramenta, essa frase vai no argumento',
    '  "thought" da propria chamada -- e ela que aparece no log do usuario ao lado',
    '  da acao. Dizer e agir juntos e o comportamento esperado, nao opcional:',
    '  sem a frase nao ha como saber o que voce pretendeu escrever.',
    '- Resolva o calculo voce mesmo.',
    '- Use "build_expression" para fracao, potencia, raiz, equacao ou "solve for x".',
    '- Se a questao tem VARIOS campos, preencha TODOS numa unica acao: um item em',
    '  "inputs" para CADA campo (ex.: {"inputs":[{"index":0,"text":"a"},',
    '  {"index":1,"text":"b"},{"index":2,"text":"c"}]}). Verificar com campo vazio',
    '  marca a resposta incompleta como ERRADA, entao nao va aos poucos.',
    '- Depois de responder, o proximo passo e "click_check".',
    '- Se nao souber responder com certeza, use "ask_user" em vez de chutar.',
    '',
    'Sobre multipla escolha, leia com atencao:',
    '- O indice em [0], [1], [2] e POSICIONAL: e a ordem em que a alternativa',
    '  aparece na tela. Confie no indice, nao na letra (A, B, C) do botao.',
    '- As alternativas vem em bloco separado e NAO se repetem no ENUNCIADO.',
    '  Se o mesmo texto parecer duas vezes, use so a secao ALTERNATIVAS.',
    '- Compare as alternativas entre si antes de escolher: varias costumam ser',
    '  quase iguais e a diferenca esta num sinal, num expoente ou num limite.',
    '- Depois de decidir, reabra a lista ALTERNATIVAS e confira que o texto que',
    '  voce escrever em choice_text e mesmo o da alternativa do choice_index.',
    '  Errar o indice e o erro mais comum: acertar o raciocinio e marcar a',
    '  alternativa errada nao conta como resposta certa.',
    '- Se o bloco "*** ATENCAO" aparecer, a questao depende de um grafico ou',
    '  imagem que voce NAO esta vendo. Nao da para resolver pelo texto: use',
    '  "ask_user" e peca para o usuario olhar a questao, em vez de chutar.',
    '',
    'Sobre graficos e imagens (leia antes de desistir de qualquer figura):',
    '- Se o bloco "*** IMAGEM ANEXADA" aparecer, a figura da questao esta NA',
    '  SUA mensagem e voce esta vendo de verdade. E o caso normal: responda',
    '  usando o que le no grafico -- valores dos eixos, formato da curva,',
    '  regioes destacadas, pontos de intersecao.',
    '- Nao repita o enunciado em voz alta e responda "nao da para saber": o',
    '  grafico esta ali precisamente para voce olhar.',
    '- "ask_user" so entra quando a imagem vier ilegivel, em branco, ou',
    '  quando a figura nao for a informacao que a questao pede.',
    '- Resolva o grafico com calma antes de escolher: leia a escala dos',
    '  eixos (eles mudam entre questoes), depois teste a alternativa contra',
    '  o que voce viu. Intervalos "crescente", "decrescente", "minimo" e',
    '  "maximo" sao os casos mais comuns.',
  ].join('\n');

  function buildUserPrompt(snap) {
    const lines = [];
    lines.push('ESTADO DA PAGINA');
    lines.push(`URL: ${location.href}`);
    lines.push(`TITULO: ${document.title}`);
    lines.push('');
    lines.push('ENUNCIADO:');
    lines.push(snap.problem);
    lines.push('');

    if (snap.choices.length) {
      lines.push(`ALTERNATIVAS (${snap.choices.length}) -- o indice e posicional:`);
      snap.choices.forEach((c, i) => {
        const t = (c.text || '').trim();
        const marca = t || (c.temImagem
          ? '(SEM TEXTO: a figura desta alternativa esta anexada abaixo)'
          : '(vazio)');
        lines.push(`  [${i}] ${marca}`);
      });
      lines.push('');
    }

    // Anexos das alternativas-imagem. O aviso existe para a IA NAO tratar o
    // texto vazio como "alternativa invalida" e cair em ask_user.
    if (snap.imagensAlternativas?.length) {
      lines.push(
        `*** ALTERNATIVAS COM IMAGEM: ${snap.imagensAlternativas.length} alternativa(s) `
        + 'desta questao nao tem texto e chegam como IMAGENS na sua mensagem, cada uma '
        + 'rotulada "IMAGEM DA ALTERNATIVA [N]". Compare as figuras entre si e escolha '
        + 'o indice da que responde ao enunciado. ***'
      );
      lines.push('');
    }

    // Questao que depende de imagem: a IA nao esta vendo o grafico, so o
    // texto. Chutar aqui e erro garantido, entao avisamos explicitamente.
    if (snap.visual) {
      if (snap.image) {
        // A imagem foi anexada de verdade: a IA esta enxergando. Avisar o
        // contrario faria ela recusar uma questao que consegue resolver, e o
        // "ask_user" viraria o comportamento padrao em grafico -- exatamente o
        // que a visao veio para eliminar.
        lines.push(
          `*** IMAGEM ANEXADA: o grafico desta questao (${snap.image.w}x${snap.image.h} `
          + `${snap.image.kind}) esta na sua mensagem, logo abaixo deste texto. `
          + 'VOCE ESTA ENXERGANDO o grafico. Leia os valores dos eixos, o formato '
          + 'da curva e as regioes marcadas, e responda com base neles. '
          + 'So use "ask_user" se a imagem vier ilegivel ou nao for o que a '
          + 'questao pede. ***'
        );
      } else {
        lines.push(
          '*** ATENCAO: esta questao depende de um GRAFICO ou IMAGEM que voce '
          + 'NAO esta vendo. O texto menciona a imagem, mas o conteudo visual nao '
          + 'foi enviado. Nao e possivel deduzir a resposta correta sem ela. '
          + 'Use "ask_user" pedindo para o usuario ver a questao, em vez de escolher '
          + 'uma alternativa ao acaso. ***'
        );
      }
      lines.push('');
    }

    if (snap.inputs.length) {
      const temMath = snap.inputs.some((f) => f.kind === 'math');
      lines.push(`CAMPOS DE RESPOSTA (${snap.inputs.length}):`);
      snap.inputs.forEach((f, i) =>
        lines.push(
          `  [${i}] ${f.kind === 'math' ? 'MATEMATICA' : 'texto'}` +
            ` ${f.visible ? 'visivel' : 'oculto'}` +
            ` placeholder="${f.placeholder}" valor_atual="${f.value}"`
        )
      );
      if (temMath) {
        lines.push('  (campo "MATEMATICA" = use build_expression com LaTeX, nunca type_answer)');
      }
      lines.push('');
    }

    lines.push(`ACOES DISPONIVEIS NESTE MOMENTO: ${acoesDisponiveis(snap).join(', ')}`);
    if (snap.progress) {
      lines.push(
        `PROGRESSO: ${snap.progress.done}/${snap.progress.total} questiones acertadas; ` +
          `questao atual (${snap.progress.index + 1}) esta "${snap.progress.state}".`
      );
    }
    if (snap.feedback) lines.push(`FEEDBACK VISIVEL NA PAGINA: ${snap.feedback}`);
    if (snap.attempts > 0) lines.push(`JA TENTOU ESTA QUESTAO ${snap.attempts} VEZ(ES).`);

    // Memoria do agente NESTA questao: o que ele ja tentou e o que aconteceu.
    //
    // Sem isto, cada tique era uma resolucao do zero: o modelo nao tinha como
    // saber QUAL alternativa acabou de errar, entao podia marcar exatamente a
    // mesma. A unica memoria era o contador "JA TENTOU N VEZES", que nao diz
    // nada sobre o que foi tentado.
    //
    // O corte por fingerprint e proposital: memoria que atravessa questoes e
    // ruido, e o modelo passa a "corrigir" uma questao usando o erro de outra.
    const tentativas = state.plano?.chave === chaveDaQuestao(snap)
      ? state.plano.tentativas
      : [];
    if (tentativas.length) {
      lines.push('');
      lines.push('O QUE VOCE JA FEZ NESTA QUESTAO (NAO repita o que falhou):');
      tentativas.forEach((t, i) => {
        // O alvo mostra O QUE foi tentado, e nao so qual ferramenta. Sem o
        // valor, a memoria diz "voce tentou responder e errou" -- e o modelo
        // pode repetir exatamente o mesmo numero.
        const alvo = (() => {
          if (t.action === 'select_choice' && t.params?.choice_index != null) {
            return ` [${t.params.choice_index}]`;
          }
          if (t.action === 'type_answer' && Array.isArray(t.params?.inputs)) {
            return ' ' + t.params.inputs
              .map((x) => `[${x.index}]="${String(x.text ?? '').slice(0, 24)}"`)
              .join(' ');
          }
          if (t.action === 'build_expression' && t.params?.latex) {
            return ` "${String(t.params.latex).slice(0, 32)}"`;
          }
          if (t.params?.expression) return ` ${String(t.params.expression).slice(0, 40)}`;
          return '';
        })();
        // O veredito e o do KHAN, nao o da mecanica: "ERRADO" vale mais para o
        // modelo do que "OK" (que so diz que o campo foi preenchido).
        const veredito = t.errou ? 'ERRADO' : t.acertou ? 'CERTO' : t.ok ? 'OK' : 'FALHOU';
        lines.push(
          `  ${i + 1}. ${t.action}${alvo} -> ${veredito}`
          + (t.detail ? `: ${String(t.detail).slice(0, 90)}` : '')
          + (t.feedback ? ` | Khan: "${String(t.feedback).slice(0, 70)}"` : '')
          + (t.thought ? ` | seu raciocinio: "${String(t.thought).slice(0, 70)}"` : '')
        );
      });
      lines.push('  Corrija a abordagem. Repetir o que ja falhou nao resolve.');
    }

    lines.push('');
    lines.push('Responda apenas com o objeto JSON da acao (ou {"actions":[...]}).');
    return lines.join('\n');
  }

  /* ===========================================================================
   * OBSERVACAO DA PAGINA
   * ======================================================================== */

  // Seletores REAIS do Khan Academy (verificados no DOM em 2026-09).
  //
  // IMPORTANTE: o Khan usa o framework Perseus e as classes "externas" sao
  // hashes de CSS-in-JS (ex.: _1yfus1e) que mudam a cada build. Por isso
  // ancoramos em data-testid / data-* e nas classes estaveis do Perseus.
  const EXERCISE_SELECTORS = [
    '[data-assessment-item-id]',   // item de avaliacao atual (o mais estavel)
    '.framework-perseus',         // form do Perseus: enunciado + resposta
    '.perseus-renderer',          // conteudo renderizado da questao
  ];

  const SEL = {
    checkBtn: '[data-testid="exercise-check-answer"]',
    // NUNCA usar [data-testid*="next"] aqui: casa com "course-next-nav", a seta
    // de navegacao do curso, e o agente passa a clicar no lugar errado.
    nextBtn: '[data-testid="exercise-next-question"]',
    skipBtn: '[data-testid="exercise-skip-button"]',
    startOver: '[data-testid="start-over-button"]',
    progress: '[data-testid^="exercise-progress-"]',
    mathKeyboardOpen: 'button[aria-label*="teclado matem" i]:not([aria-label*="fechar" i])',
    mathKeyboardClose: 'button[aria-label*="fechar teclado matem" i]',
    // Campo de matematica. Dois renders do MESMO widget:
    //   .perseus-math-input -> desktop
    //   .keypad-input       -> MOBILE (Khan troca o renderizador no celular)
    // Sem o segundo, o agente reportava "0 campos" numa questao que TEM campo:
    // no mobile o widget e um <div role="textbox"> com um .mq-root-block
    // dentro, e nao um input. Ver digitarPorEventos para como se digita nele.
    mathField: '.perseus-math-input, .keypad-input',
    mathBlock: '.perseus-widget-expression',
    mathRendered: '.mq-root-block',
    perseusRenderer: '.perseus-renderer',
  };

  // O exercicio visivel. O Khan deixa o item anterior no DOM durante a
  // transicao, entao filtrar por visibilidade e obrigatorio para nao agir
  // no no errado.
  function findExerciseRoot() {
    for (const sel of EXERCISE_SELECTORS) {
      for (const el of document.querySelectorAll(sel)) {
        if (!isVisible(el)) continue;
        // Precisa conter enunciado + algum controle de resposta.
        const hasText = (el.innerText || '').trim().length > 8;
        const hasControl =
          el.querySelector(SEL.mathField) ||
          el.querySelector('input:not([type=hidden]), textarea, [contenteditable]') ||
          el.querySelector('input[type=radio], [role=radio], [role=checkbox]');
        if (hasText && hasControl) return el;
      }
    }
    // Sem raiz confiavel: cai na raiz do Perseus, se houver.
    return document.querySelector('.framework-perseus') || null;
  }

  function isVisible(el) {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
  }

  // Texto do enunciado, limpando ruido de template do Khan.
  // Frases que indicam que a resposta depende de uma imagem que o agente
  // nao consegue ler. Sem este aviso a IA recebe "representada graficamente
  // abaixo", nao ve o grafico e chuta uma alternativa -- errando sempre.
  //
  // Sao frases INTEIRAS, e nao palavras soltas: uma lista de palavras curtas
  // ("calculo", "figura", "imagem") dispara falso positivo em perguntas de
  // texto puro -- "Calcule 9m+4" foi marcado como questao visual.
  const VISUAL_PHRASES = [
    // "grafico" SOZINHO ja basta. As regras abaixo exigiam o verbo logo depois
    // ("grafico mostra") e por isso "O grafico de setores mostra os percentuais
    // de participacao..." NAO casava: a questao era tratada como texto, a
    // imagem nunca era capturada, e a IA respondia "nao vejo o grafico".
    //
    // Se a questao cita um grafico e a captura falhar, o prompt avisa a IA de
    // que a figura nao veio -- que e o comportamento certo, e melhor do que
    // fingir que a questao e de texto.
    /\bgr[áa]fic/,
    /graficamente/,
    /representad[ao]\s+(n[ao]\s+)?gr[áa]fic/,
    /gr[áa]fic\w*\s+(abaixo|acima|ao\s+lado|à\s+seguir)/,
    /(abaixo|acima|à\s+seguir|ao\s+lado)[^.]*\bgr[áa]fic/,
    /(observe|veja|olhe|analise|considere)\s+(o\s+|a\s+)?gr[áa]fic/,
    /(a\s+partir|pelo|conforme)\s+(d[oa]\s+|do\s+)?gr[áa]fic/,
    /\bgr[áa]fic\w*\s+(mostrado|mostrada|abaixo|representado|representada)/,
    // A forma de presente do verbo faltava, e e a mais comum do enunciado:
    // "O grafico mostra a variacao de velocidade..." nao casava com nenhum
    // padrao acima, a questao era tratada como texto, a imagem nao era
    // anexada e a IA caia em ask_user mesmo tendo visao.
    /\bgr[áa]fic\w*\s+(mostra|mostram|mosstra|mosstram|indica|indicam|aparece|aparecem|representa|representam|ilustra|ilustram|exibe|exibem|descreve|descrevem|apresenta|apresentam|contraste|resume|detalha)/,
    /\b(imagem|figura|diagrama|fotonometria|mapa)\b(?![^.]{0,40}\b(valor|area|medida|total)\b)/,
    /na\s+imagem\s+(abaixo|acima|mostrada|representada|citada)/,
    // Mesma lacuna com "a imagem mostra": so havia as formas participiais.
    /\b(imagem|figura)\s+(mostra|mostram|indica|indicam|aparece|representa|ilustra|exibe|apresenta)/,
    /como\s+(mostrado|representado|ilustrado)\s+(abaixo|acima)/,
    /(qual|quais)\s+(gr[áa]fic|figura|imagem)/,
    /descreva\s+(o|a)\s+(gr[áa]fic|figura|imagem)/,
  ];

  // Blocos de "conteudo relacionado" / rodape de video. Ficam dentro da raiz do
  // exercicio, mas nao fazem parte do enunciado: a miniatura de "Video 2
  // minutos" e um <img> de 96x54 sem alt e fazia TODA questao de texto ser
  // tratada como visual.
  const OFF_TOPIC = [
    'conteudo relacionado', 'conteúdo relacionado', 'related content',
    'watch video', 'assista ao video', 'assista ao vídeo', 'video ',
    'transcricao', 'transcrição', 'key facts', 'fatos-chave', 'deep dive',
  ].join(', ');

  // O rodape vem como texto solto ("Conteudo relacionado", "Video 2
  // minutos", "Calculo de expressoes com uma variavel"), nao como elemento
  // com classe estavel -- entao ele e cortado por texto, a partir do titulo
  // do bloco ate o fim.
  function stripRelatedFooter(text) {
    const i = text.search(
      /(conte[uú]do relacionado|related content|assista ao v[ií]deo|watch video|transcri[çc][aã]o|key facts|fatos-chave)/i
    );
    return i === -1 ? text : text.slice(0, i).trim();
  }

    /* ===========================================================================
   * VISAO: MANDAR O GRAFICO PARA A IA
   * ======================================================================== */
  // Sem imagem, uma questao de grafico e irrespondivel: o texto so diz "o
  // grafico abaixo" e a IA nao tem o que olhar, entao chuta. Modelos
  // multimodais (qwen3.8-27b no Groq, gpt-4o, llama-3.2-vision) aceitam a
  // imagem como parte da mensagem em content[].type == "image_url".
  //
  // Como o grafico do Khan sai da pagina (medido no grafico real de
  // "comportamento-de-uma-funcao-real-crescimento"):
  //
  //   1. div.graphie NAO tem o desenho. E o SVG vazio do Raphael mais os
  //      rotulos MathJax dos eixos. Rasterizar isso da PNG branco de 1 KB,
  //      e a IA responde "a imagem veio em branco".
  //   2. O desenho esta numa <img class="image-loader-img"> que aponta para
  //      cdn.kastatic.org/ka-perseus-graphie/<hash>.svg.
  //   3. Passar essa URL ao provider nao resolve: o Groq aceita apenas
  //      PNG/JPEG/WebP e responde "invalid image data" para image/svg+xml.
  //   4. fetch(src, {credentials:'include'}) falha com "Failed to fetch": o
  //      CDN responde Access-Control-Allow-Origin: * e credenciais junto com
  //      wildcard sao proibidas. O mesmo fetch SEM credentials devolve 200.
  //   5. Desenhar a <img> cross-origin num canvas nao da: canvas tainted.
  //
  // O caminho que funciona e, por isso, o implementado aqui: achar a <img>,
  // baixar o ARQUIVO como texto com fetch simples, rasterizar esse texto num
  // canvas local e embutir o resultado como JPEG data URL. Os bytes vieram do
  // fetch, entao o canvas nao fica tainted.
  const VISION_MAX_PX = 1200;     // lado maior maximo da imagem enviada
  const VISION_MIN_PX = 720;      // lado maior MINIMO ao rasterizar vetor
  const VISION_QUALITY = 0.88;    // qualidade do JPEG (0 a 1)

  // Escala de renderizacao de uma figura VETORIAL.
  //
  // Antes so encolhia (Math.min(1, MAX/lado)). Um SVG do Khan sem width/height
  // declara naturalWidth 300 por padrao, entao o grafico era rasterizado a
  // 300px e os numeros do eixo chegavam borrados -- o modelo lia o formato da
  // curva e errava a escala, que e justamente o que a questao pergunta.
  //
  // Vetor NAO perde informacao ao ampliar: renderizar a 720px produz de fato
  // mais detalhe. Por isso ha um MINIMO, e nao so um teto.
  //
  // Isto vale so para SVG. Em PNG/JPEG (raster) ampliar nao cria informacao --
  // e esse caminho nem passa por aqui.
  function escalaDeRender(w, h) {
    const lado = Math.max(1, w, h);
    return Math.min(VISION_MAX_PX, Math.max(VISION_MIN_PX, lado)) / lado;
  }

  // Candidatos a "figura da questao", em ordem de confianca. A <img> do CDN
  // vem primeiro porque e a unica que carrega o desenho real.
  const IMG_SEL = [
    'img.image-loader-img',
    'img.khan-image',
    '.svg-image img',
    '.graphie-container img',
    '[data-testid="graph"] img',
    'img',
  ];

  function substantial(el) {
    if (!el || !isVisible(el)) return false;
    const r = el.getBoundingClientRect();
    return r.width >= 60 && r.height >= 40;
  }

  // A figura que vale a pena olhar: grande, visivel, dentro do enunciado e
  // com src utilizavel.
  function findVisualElement(root) {
    const vistos = new Set();
    for (const sel of IMG_SEL) {
      for (const el of root.querySelectorAll(sel)) {
        if (vistos.has(el) || !substantial(el)) continue;
        vistos.add(el);
        // Fora do enunciado: rodape de conteudo relacionado, dicas, rodape.
        if (el.closest(OFF_TOPIC)) continue;
        if (el.closest('.related-content, .site-footer, .footer')) continue;
        if (!(el.getAttribute('src') || '').trim()) continue;
        return el;
      }
    }
    return null;
  }

  // Baixa o arquivo da <img>. O modo cors e explicito porque o default
  // ("cors", mas sem credenciais) ja serve -- o que quebrava era passar
  // credentials: 'include' junto com o Access-Control-Allow-Origin: *.
  async function fetchImage(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status} ao baixar a figura`);
    return {
      tipo: (r.headers.get('content-type') || '').toLowerCase(),
      bytes: await r.arrayBuffer(),
    };
  }

  function bytesToBase64(buf) {
    const bytes = new Uint8Array(buf);
    let s = '';
    // Em pedacos de 8 KB: concatenar tudo de uma vez estoura o limite de
    // argumento do apply() e quebra em imagens grandes.
    for (let i = 0; i < bytes.length; i += 8192) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
    }
    return btoa(s);
  }

  function looksLikeSvg(tipo, bytes) {
    if (/svg/.test(tipo)) return true;
    // Alguns CDNs mandam octet-stream; o(bytes) revela o "<svg" no inicio.
    const cab = new TextDecoder().decode(new Uint8Array(bytes.slice(0, 200))).trim();
    return /^<(\?xml|svg)/i.test(cab);
  }

  // Rasteriza markup SVG e devolve JPEG data URL.
  //
  // Duas tentativas, e a segunda so acontece se a primeira falhar. O motivo foi
  // MEDIDO, nao suposto: o blob: contamina o canvas no Chrome quando o SVG
  // contem <foreignObject> -- e e exatamente o caso do print da area da questao
  // ("Tainted canvases may not be exported"). Com data: URL o mesmo SVG passa
  // limpo nos dois motores:
  //
  //   Chromium:  blob: contaminou=true    data: contaminou=false
  //   Firefox:   blob: contaminou=false   data: contaminou=false
  //
  // O data: fica como SEGUNDA tentativa porque encodeURIComponent infla o texto
  // em ~3x, e no caminho normal (o grafico da CDN) o SVG e grande.
  function rasterizarSvg(url, revogar) {
    return new Promise((resolve) => {
      const probe = new Image();
      probe.onload = () => {
        try {
          const w = Math.max(1, probe.naturalWidth || 300);
          const h = Math.max(1, probe.naturalHeight || 300);
          const escala = escalaDeRender(w, h);
          const W = Math.max(1, Math.round(w * escala));
          const H = Math.max(1, Math.round(h * escala));
          const c = document.createElement('canvas');
          c.width = W; c.height = H;
          const g = c.getContext('2d');
          // Fundo branco explicito: SVG transparente sobre canvas
          // transparente vira JPEG com fundo preto, e o eixo some.
          g.fillStyle = '#ffffff';
          g.fillRect(0, 0, W, H);
          g.drawImage(probe, 0, 0, W, H);
          resolve({ dataUrl: c.toDataURL('image/jpeg', VISION_QUALITY), w: W, h: H });
        } catch (e) {
          // Guardar o motivo importa: "nao consegui capturar a imagem" sem dizer
          // por que obriga a investigar de novo toda vez.
          state.visionWhy = `canvas recusou a imagem (${e.message})`;
          resolve(null);
        } finally {
          if (revogar) URL.revokeObjectURL(url);
        }
      };
      probe.onerror = () => {
        if (revogar) URL.revokeObjectURL(url);
        state.visionWhy = 'o navegador recusou renderizar o SVG';
        resolve(null);
      };
      probe.src = url;
    });
  }

  async function svgToJpeg(markup) {
    const blobUrl = URL.createObjectURL(new Blob([markup], { type: 'image/svg+xml;charset=utf-8' }));
    const viaBlob = await rasterizarSvg(blobUrl, true);
    if (viaBlob) return viaBlob;
    return rasterizarSvg(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`, false);
  }

  // Props de SVG que valem a pena achatar. So o que o Khan usa de verdade.
  const SVG_STYLE_PROPS = [
    'fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity',
    'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin', 'opacity',
    'font-family', 'font-size', 'font-weight', 'text-anchor',
    'dominant-baseline', 'stop-color', 'stop-opacity',
  ];

  // Aplica os estilos computados como atributo em cada elemento.
  //
  // necessario quando o SVG vem do DOM e depende de CSS do site: serializado
  // sozinho, sai tudo com o estilo padrao (preenchimento preto) e a IA recebe
  // um borrão sem informação. Por atributo, ganha sem depender de
  // especificidade de CSS. Quando o SVG ja vem do CDN, ele e autossuficiente
  // e essa funcao nem e chamada.
  function inlineSvgStyles(svg) {
    const partes = [];
    for (const el of svg.querySelectorAll('*')) {
      const cs = getComputedStyle(el);
      const decl = [];
      for (const p of SVG_STYLE_PROPS) {
        const v = cs.getPropertyValue(p);
        if (v && v !== 'normal' && v !== 'none' && v !== 'auto') decl.push(`${p}:${v}`);
      }
      if (decl.length) partes.push(`${el.tagName}{${decl.join(';')}}`);
    }
    return partes.join('');
  }

  // Reserva: figura desenhada direto no DOM, sem <img> espelho. O Khan
  // antigo e varias figuras de outros sites funcionam assim.
  //
  // O que nao funciona e o div.graphie do grafico atual: ele contem um SVG do
  // Raphael so com <desc> e <defs>, mais os rotulos dos eixos. Por isso o
  // filtro exige formas de verdade E area util -- sem isso cai no texto e a
  // imagem sai em branco.
  async function svgDoDomParaJpeg(root) {
    const svgs = [...root.querySelectorAll('svg')].filter((s) => {
      if (!substantial(s)) return false;
      // Formula tambem e SVG (MathJax/MathQuill desenham com path). Sem esta
      // exclusao, uma questao de texto com formula era capturada como "grafico"
      // e o prompt passava a dizer "voce esta vendo o grafico" para a IA.
      if (s.closest(SEL_MATH_SVG)) return false;
      const formas = s.querySelectorAll(
        'path, polyline, polygon, circle, ellipse, rect, line, text, image, use'
      );
      if (formas.length < 4) return false;
      const r = s.getBoundingClientRect();
      return r.width * r.height >= 60 * 40;
    });
    if (!svgs.length) return null;

    // O maior desenho ganha: e o grafico, nao um icone ou um simbolo.
    svgs.sort((a, b) => {
      const ra = a.getBoundingClientRect();
      const rb = b.getBoundingClientRect();
      return rb.width * rb.height - ra.width * ra.height;
    });
    const svg = svgs[0];

    const r = svg.getBoundingClientRect();
    const vb = (svg.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number);
    const w = Math.max(1, Math.round(vb[2] || r.width || 300));
    const h = Math.max(1, Math.round(vb[3] || r.height || 300));
    const escala = escalaDeRender(w, h);
    const W = Math.round(w * escala);
    const H = Math.round(h * escala);

    let markup = new XMLSerializer().serializeToString(svg);
    markup = markup.replace(/<svg([^>]*)>/, (m, attrs) => {
      let a = attrs;
      if (!/xmlns=/.test(a)) a += ' xmlns="http://www.w3.org/2000/svg"';
      // width="100%" nao serve como dimensao intrinsica: o <img> de teste
      // fica com naturalWidth 0 e o canvas sai em branco. Vira pixel.
      a = a.replace(/\bwidth="[^"]*"/, `width="${W}"`)
           .replace(/\bheight="[^"]*"/, `height="${H}"`);
      if (!/\bwidth=/.test(a)) a += ` width="${W}"`;
      if (!/\bheight=/.test(a)) a += ` height="${H}"`;
      return `<svg${a}>`;
    });
    const estilo = inlineSvgStyles(svg);
    if (estilo) {
      markup = markup.replace(/(<svg[^>]*>)/, `$1<style type="text/css"><![CDATA[${estilo}]]></style>`);
    }

    const jpeg = await svgToJpeg(markup);
    return jpeg ? { dataUrl: jpeg.dataUrl, w: jpeg.w, h: jpeg.h } : null;
  }

  /* ===========================================================================
   * PRINT DA AREA (figura que nao e <img> nem um unico <svg>)
   * ---------------------------------------------------------------------------
   * Nao existe API no navegador para fotografar um pedaco da pagina de dentro do
   * proprio script -- getDisplayMedia exigiria o usuario escolher a tela a cada
   * vez, e o agente e autonomo. O caminho viavel e serializar o DOM num SVG com
   * <foreignObject> e rasterizar esse SVG: o navegador desenha o HTML ali dentro.
   *
   * Limites, ditos em vez de escondidos:
   *   - CSS de FOLHA EXTERNA nao se aplica dentro do foreignObject, entao os
   *     estilos computados sao copiados para style="" antes de serializar;
   *   - <canvas> nao e serializavel: vira <img> com o bitmap atual;
   *   - <img> de outra origem deixam o canvas "sujo" e o toDataURL lanca. Elas
   *     viram data URL quando da, e sao DESCARTADAS quando nao da -- melhor uma
   *     figura sem um pedaco do que nenhuma figura.
   *
   * Por isso ele e o ULTIMO recurso: o <img> da CDN e o SVG do DOM saem
   * melhores. Este existe para o que sobra (grafico desenhado com CSS, por
   * exemplo).
   * ======================================================================== */
  async function screenshotDaArea(el, maxLado = VISION_MAX_PX) {
    const r = el.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width));
    const h = Math.max(1, Math.round(r.height));
    if (w < 40 || h < 30) return null;

    const clone = el.cloneNode(true);
    const originais = [el, ...el.querySelectorAll('*')];
    let clones = [clone, ...clone.querySelectorAll('*')];

    // (1) <canvas> -> <img> com o bitmap atual (canvas nao serializa).
    for (let i = 0; i < originais.length && i < clones.length; i++) {
      if (originais[i].tagName !== 'CANVAS') continue;
      try {
        const img = document.createElement('img');
        img.setAttribute('src', originais[i].toDataURL('image/png'));
        clones[i].replaceWith(img);
      } catch { /* canvas sem permissao: fica em branco, mas nao derruba */ }
    }

    // (2) Estilos computados -> style="".
    //
    // O cap existe porque o exercicio inteiro passa de mil elementos e copiar
    // estilo de todos trava a thread por um instante visivel.
    const ESTILO = [
      'display', 'position', 'top', 'left', 'right', 'bottom', 'float',
      'color', 'background-color', 'background-image', 'background-size',
      'font-family', 'font-size', 'font-weight', 'font-style', 'line-height',
      'text-align', 'text-decoration', 'white-space',
      'border', 'border-radius', 'padding', 'margin',
      'width', 'height', 'max-width', 'opacity', 'overflow', 'box-shadow',
      'fill', 'stroke', 'stroke-width', 'stroke-dasharray',
      'transform', 'transform-origin', 'vertical-align',
      'justify-content', 'align-items', 'flex-direction', 'flex-wrap', 'gap',
    ];
    clones = [clone, ...clone.querySelectorAll('*')];
    for (let i = 0; i < clones.length && i < 1500; i++) {
      const c = clones[i];
      if (!c || c.nodeType !== 1 || !originais[i]) continue;
      try {
        const cs = getComputedStyle(originais[i]);
        let decl = '';
        for (const p of ESTILO) {
          const v = cs.getPropertyValue(p);
          if (v && v !== 'none' && v !== 'normal' && v !== 'auto') decl += `${p}:${v};`;
        }
        if (decl) c.setAttribute('style', decl + (c.getAttribute('style') || ''));
      } catch { /* elemento sem layout */ }
    }

    // (3) <img> -> data URL, para nao sujar o canvas.
    for (const ic of [...clone.querySelectorAll('img')]) {
      const src = ic.getAttribute('src') || '';
      if (!src || /^data:/i.test(src)) continue;
      try {
        const { tipo, bytes } = await fetchImage(src);
        const mime = /^image\/(png|jpe?g|webp|gif)/.test(tipo) ? tipo : 'image/png';
        ic.setAttribute('src', `data:${mime};base64,${bytesToBase64(bytes)}`);
      } catch { ic.remove(); }
    }

    const markup = new XMLSerializer().serializeToString(clone);
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">`
      + `<foreignObject x="0" y="0" width="${w}" height="${h}">`
      + `<div xmlns="http://www.w3.org/1999/xhtml" style="width:${w}px;height:${h}px;overflow:hidden;background:#ffffff">`
      + `${markup}</div></foreignObject></svg>`;

    const jpeg = await svgToJpeg(svg);
    return jpeg ? { dataUrl: jpeg.dataUrl, w: jpeg.w, h: jpeg.h } : null;
  }

  // Reencoda uma imagem RASTER grande para JPEG menor.
  //
  // O CDN do Khan as vezes manda um PNG grande -- 1,1 MB num caso real. Isso
  // vai inteiro no corpo da requisicao: a chamada fica lenta, cara em tokens de
  // visao e sujeita a timeout (o log mostrou "A IA ainda nao respondeu (4s)"
  // com 1556 KB de corpo). Reencodar num canvas resolve sem perder o que a IA
  // precisa ler.
  //
  // So age acima do limiar: reencodar uma imagem que ja e pequena perderia
  // qualidade de graca.
  async function reencodarRaster(dataUrl, limiarKb = 150) {
    const kb = Math.round((dataUrl.length * 3) / 4 / 1024);
    if (kb <= limiarKb) return { dataUrl, kb, reencodou: false };

    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        try {
          const w = Math.max(1, img.naturalWidth || 300);
          const h = Math.max(1, img.naturalHeight || 300);
          const escala = escalaDeRender(w, h);
          const W = Math.max(1, Math.round(w * escala));
          const H = Math.max(1, Math.round(h * escala));
          const c = document.createElement('canvas');
          c.width = W; c.height = H;
          const g = c.getContext('2d');
          // Fundo branco: PNG com transparencia viraria fundo preto no JPEG.
          g.fillStyle = '#ffffff';
          g.fillRect(0, 0, W, H);
          g.drawImage(img, 0, 0, W, H);
          const saida = c.toDataURL('image/jpeg', VISION_QUALITY);
          resolve({
            dataUrl: saida,
            kb: Math.round((saida.length * 3) / 4 / 1024),
            reencodou: true,
            w: W,
            h: H,
          });
        } catch {
          resolve({ dataUrl, kb, reencodou: false });
        }
      };
      img.onerror = () => resolve({ dataUrl, kb, reencodou: false });
      img.src = dataUrl;
    });
  }

  // Devolve {dataUrl, w, h, kind} para a IA, ou null se nao houver figura.  // Nunca lanca: falhar na captura e comum (figura carregando, CDN fora) e
  // nao pode derrubar o loop. O motivo fica em state.visionWhy para o log.
  async function captureVisual(root) {
    if (!state.cfg.vision) return null;
    const el = findVisualElement(root);
    if (!el) {
      // Sem <img>, tenta a reserva do SVG desenhado no proprio DOM. E o caso do
      // grafico de setores, que e <svg> com <path> e nao um <svg><image>.
      const dom = await svgDoDomParaJpeg(root);
      if (dom) return { ...dom, kind: 'grafico vetorial do DOM' };

      // Ultimo recurso: o "print" da area da questao. Cobre o que nao e nem
      // <img> nem um unico <svg> (grafico desenhado com CSS, por exemplo).
      const shot = await screenshotDaArea(root);
      if (shot) return { ...shot, kind: 'print da area da questao' };

      // Nao sobrescrever um motivo ja registrado: svgToJpeg explica POR QUE o
      // navegador recusou a imagem, e essa e a informacao util para depurar.
      if (!state.visionWhy) state.visionWhy = 'nenhuma figura grande encontrada no enunciado';
      return null;
    }

    const src = (el.getAttribute('src') || '').trim();
    const rect = el.getBoundingClientRect();
    const w = Math.round(rect.width) || el.naturalWidth || 0;
    const h = Math.round(rect.height) || el.naturalHeight || 0;

    try {
      // (1) data URL ja embutida na pagina.
      if (/^data:/i.test(src)) {
        const ehSvg = /^data:image\/svg/i.test(src);
        if (!ehSvg) return { dataUrl: src, w, h, kind: 'imagem embutida' };
        const corpo = src.replace(/^data:image\/svg\+xml[^,]*,/, '');
        const markup = /;base64,/i.test(src)
          ? new TextDecoder().decode(Uint8Array.from(atob(corpo), (c) => c.charCodeAt(0)))
          : decodeURIComponent(corpo);
        const jpeg = await svgToJpeg(markup);
        if (jpeg) return { dataUrl: jpeg.dataUrl, w: jpeg.w, h: jpeg.h, kind: 'grafico vetorial' };
        state.visionWhy = 'SVG embutido nao renderizou';
        return null;
      }

      // (2) Baixa e converte. E o caminho do grafico do Khan.
      const { tipo, bytes } = await fetchImage(src);
      if (looksLikeSvg(tipo, bytes)) {
        const markup = new TextDecoder().decode(bytes);
        const jpeg = await svgToJpeg(markup);
        if (jpeg) return { dataUrl: jpeg.dataUrl, w: jpeg.w, h: jpeg.h, kind: 'grafico vetorial' };
        state.visionWhy = 'SVG baixado, mas nao renderizou no canvas';
        return null;
      }
      // (3) Ja era PNG/JPEG: base64 dos bytes, com o MIME que o CDN mandou.
      const mime = /^image\/(png|jpe?g|webp|gif)/.test(tipo) ? tipo : 'image/png';
      const bruto = `data:${mime};base64,${bytesToBase64(bytes)}`;
      // PNG grande do CDN vira JPEG: o corpo da requisicao chegava a 1,5 MB.
      const r = await reencodarRaster(bruto);
      if (r.reencodou) {
        log(`Figura reencodada para JPEG: ${r.kb} KB (era ${Math.round((bruto.length * 3) / 4 / 1024)} KB).`, 'info');
      }
      return {
        dataUrl: r.dataUrl,
        w: r.w || w,
        h: r.h || h,
        kind: r.reencodou ? 'imagem (reencodada)' : 'imagem',
      };
    } catch (e) {
      // Uma falha aqui nao e nova tentativa de outro caminho: e a mesma
      // figura, e repetir so queima cota. O agente segue com o texto e
      // mantem o aviso de "voce nao esta vendo a imagem".
      state.visionWhy = e.message;
      return null;
    }
  }

  /* ===========================================================================
   * ALTERNATIVAS QUE SAO IMAGEM
   * ---------------------------------------------------------------------------
   * "Qual destes graficos representa ...?" traz alternativas sem NENHUM texto:
   * o choiceText volta vazio e a IA recebia "[0] ", "[1] ", "[2] " -- escolher
   * era chute. Aqui cada alternativa sem texto e enviada como uma imagem
   * rotulada com o seu indice.
   *
   * Enviamos SO as que nao tem texto: alternativa com texto ja e legivel, e
   * anexar imagem dela gastaria tokens de visao sem informacao nova.
   * ======================================================================== */
  async function capturarFiguraDe(node) {
    // (1) <img> dentro da alternativa -- o caso comum.
    const img = [...node.querySelectorAll('img')]
      .find((e) => (e.getAttribute('src') || '').trim());
    if (img) {
      const src = img.getAttribute('src').trim();
      try {
        if (/^data:/i.test(src)) {
          if (!/^data:image\/svg/i.test(src)) return { dataUrl: src };
          const markup = src.replace(/^data:image\/svg\+xml[^,]*,/, '');
          const j = await svgToJpeg(markup);
          return j ? { dataUrl: j.dataUrl } : null;
        }
        const { tipo, bytes } = await fetchImage(src);
        if (looksLikeSvg(tipo, bytes)) {
          const j = await svgToJpeg(new TextDecoder().decode(bytes));
          return j ? { dataUrl: j.dataUrl } : null;
        }
        const mime = /^image\/(png|jpe?g|webp|gif)/.test(tipo) ? tipo : 'image/png';
        return { dataUrl: `data:${mime};base64,${bytesToBase64(bytes)}` };
      } catch { /* sem rede para a figura: tenta o SVG do DOM abaixo */ }
    }
    // (2) Figura desenhada no DOM, sem <img> espelho.
    const dom = await svgDoDomParaJpeg(node);
    return dom ? { dataUrl: dom.dataUrl } : null;
  }

  async function capturarAlternativasSemTexto(root, choices) {
    if (!state.cfg.vision || state.visionDead || state.extrasDead) return [];
    const nodes = choiceNodes(root);
    const saida = [];
    // Teto de 6: uma questao com 10 alternativas-imagem viraria uma mensagem
    // enorme, e o custo de visao cresce por imagem.
    for (let i = 0; i < choices.length && saida.length < 6; i++) {
      const c = choices[i];
      if ((c.text || '').trim() || !c.temImagem) continue;
      const node = nodes[i];
      if (!node) continue;
      const fig = await capturarFiguraDe(node);
      if (fig) saida.push({ index: i, dataUrl: fig.dataUrl, rotulo: `IMAGEM DA ALTERNATIVA [${i}]:` });
    }
    return saida;
  }

  // SVG que e GRAFICO, e nao formula.
  //
  // MathJax e MathQuill desenham formula com <path> tambem. Confundir os dois e
  // caro: uma formula anexada como "grafico" faz o prompt dizer "voce esta
  // vendo o grafico" para uma questao de texto puro.
  const SEL_MATH_SVG = 'mjx-container, .MathJax, .mathjax-wrapper, .mq-root-block, .mq-math-mode, .perseus-math-input, .keypad-input';
  function ehSvgDeGrafico(svg) {
    if (!svg || !substantial(svg)) return false;
    if (svg.closest(SEL_MATH_SVG)) return false;
    return svg.querySelectorAll('path, polyline, polygon, circle, ellipse, rect, line').length >= 3;
  }

  function looksVisual(root, problem) {
    // NAO clonar a raiz aqui.
    //
    // O clone fica destacado do documento, e elemento destacado tem retangulo
    // 0x0 -- entao TODO teste de tamanho falhava e a checagem de imagem nunca
    // passava. Na pratica a decisao caia so nas frases do enunciado, e uma
    // questao com figura cujo texto nao casasse com nenhuma frase era tratada
    // como texto puro: a imagem nunca era capturada e a IA respondia "nao vejo
    // o grafico".
    const foraDoEnunciado = (el) =>
      !!el.closest(OFF_TOPIC) || !!el.closest('.related-content, .site-footer, .footer');
    const isSubstancial = (el) => {
      const r = el.getBoundingClientRect();
      return isVisible(el) && r.width >= 60 && r.height >= 40;
    };

    const imgs = Array.from(
      root.querySelectorAll('img, canvas, video, svg image, [data-testid*="graph"]')
    ).filter((el) => isSubstancial(el) && !foraDoEnunciado(el));

    if (imgs.length) {
      // Alguma imagem com alt que descreve o conteudo = o agente le o alt.
      const comAlt = imgs.filter((i) => {
        const alt = (i.getAttribute('alt') || '').trim();
        return alt && !/^(imagem(\s+de)?|image|photo|picture|untitled|gr[áa]fico|graph|v[íi]deo|video)\.?$/i.test(alt);
      });
      if (!comAlt.length) return true;
    }

    // SVG DESENHADO NO DOM. Um grafico de setores e <svg> com <path>, e nao um
    // <svg><image>: sem esta checagem a questao era tratada como texto.
    if (Array.from(root.querySelectorAll('svg')).some((s) => ehSvgDeGrafico(s) && !foraDoEnunciado(s))) {
      return true;
    }

    const t = (problem || '').toLowerCase();
    return VISUAL_PHRASES.some((re) => re.test(t));
  }

  /* ===========================================================================
   * MATEMATICA NOS ENUNCIADOS (MathJax / KaTeX)
   * ======================================================================== */
  // O enunciado e renderizado em MathJax: a fracao e <mjx-mfrac> com
  // <mjx-num>9</mjx-num><mjx-den>m</mjx-den>. Ler com innerText concatena
  // numerador e denominador e produz "9m + 4" para 9/m + 4 -- sem barra, sem
  // erro visivel. A IA calcula 9*3+4 = 31 com toda a aparencia de estar
  // certa, e o agente trava numa questao cujo enunciado ela nunca leu de fato.
  //
  // A correcao e serializar a arvore matematica, nao o texto. O MathJax v3
  // sempre embute MathML assistivo (<mjx-assistive-mml>), que ja vem na forma
  // canonica; o CHTEL e so a reserva.

  // MathML -> texto. Cobre o que aparece em algebra do secundario.
  function mathmlToText(node) {
    if (!node) return '';
    const kids = Array.from(node.children || []);
    const tag = (node.nodeName || '').toLowerCase();

    switch (tag) {
      case 'math':
      case 'mrow':
      case 'mstyle':
      case 'mpadded':
      case 'mphantom':
        return kids.map(mathmlToText).join('');

      // <semantics> guarda a expressao E o LaTeX em <annotation>; juntar os
      // dois duplicaria a formula.
      case 'semantics':
      case 'mfenced':
        return kids.filter((k) => k.localName !== 'annotation').map(mathmlToText).join('');

      case 'mi': case 'mn': case 'ms': case 'mtext':
        return node.textContent || '';

      case 'mo': {
        const t = (node.textContent || '').trim();
        // Espaco ao redor do operador: sem isso "9/m+4" fica ambiguo, e a IA
        // pode ler "9/(m+4)".
        return t ? ` ${t} ` : ' ';
      }
      case 'mspace':
        return ' ';

      case 'mfrac': {
        // MathML Core: os dois filhos vem direto. MathML de apresentacao
        // antigo: vem envolvidos em <num>/<den>.
        const numEl = kids.find((k) => k.localName === 'num');
        const denEl = kids.find((k) => k.localName === 'den');
        const num = numEl ? mathmlToText(numEl) : mathmlToText(kids[0]);
        const den = denEl ? mathmlToText(denEl) : mathmlToText(kids[1]);
        return `(${num.trim()})/(${den.trim()})`;
      }
      case 'msqrt':
        return `sqrt(${kids.map(mathmlToText).join('').trim()})`;
      case 'mroot': {
        const base = mathmlToText(kids[0]).trim();
        const idx = mathmlToText(kids[1]).trim();
        return `(${base})^(1/(${idx}))`;
      }
      case 'msup': return `${mathmlToText(kids[0]).trim()}^(${mathmlToText(kids[1]).trim()})`;
      case 'msub': return `${mathmlToText(kids[0]).trim()}_(${mathmlToText(kids[1]).trim()})`;
      case 'msubsup': {
        const base = mathmlToText(kids[0]).trim();
        return `${base}_(${mathmlToText(kids[1]).trim()})^(${mathmlToText(kids[2]).trim()})`;
      }
      case 'munder': return `${mathmlToText(kids[0]).trim()}_(${mathmlToText(kids[1]).trim()})`;
      case 'mover': return `${mathmlToText(kids[0]).trim()}^(${mathmlToText(kids[1]).trim()})`;

      case 'mtable': case 'mtr': case 'mtd':
        return ' ' + kids.map(mathmlToText).join(' | ') + ' ';

      default:
        return node.textContent || '';
    }
  }

  // CHTEL -> texto. Reserva para quando nao ha MathML. Trata o mfrac, que e o
  // caso que produz resposta errada; o resto cai no texto dos glifos.
  function chtmlToText(node) {
    const kids = Array.from(node.children || []);
    const cls = (node.getAttribute('class') || '').split(/\s+/);
    const has = (c) => cls.includes(c);

    if (has('mjx-mfrac')) {
      const num = kids.find((k) => (k.getAttribute('class') || '').split(/\s+/).includes('mjx-num'));
      const den = kids.find((k) => (k.getAttribute('class') || '').split(/\s+/).includes('mjx-den'));
      return `(${chtmlToText(num)})/(${chtmlToText(den)})`;
    }
    if (has('mjx-msqrt')) return `sqrt(${kids.map(chtmlToText).join('')})`;
    // mjx-c e cada glifo isolado do numero de referencia do MathJax: o
    // textContent dele ja e a letra renderizada, entao concatenar funciona.
    if (has('mjx-c')) return node.textContent || '';
    // Estrutura de apoio da fracao: nao contribute para o texto.
    if (has('mjx-num') || has('mjx-den') || has('mjx-dbox') || has('mjx-line')) return '';
    if (!kids.length) return node.textContent || '';
    return kids.map(chtmlToText).join('');
  }

  // Substitui cada bloco de matematica por um no de texto. Roda no clone
  // ANTES da remocao dos [aria-hidden], porque o mjx-math -- que traz os
  // glifos -- e aria-hidden, e o MathML assistivo esta dentro dele.
  function renderMathInPlace(root) {
    const serialize = (el) => {
      const mm = el.querySelector('mjx-assistive-mml');
      if (mm) {
        const t = mathmlToText(mm.querySelector('math') || mm);
        if (t.trim()) return t;
      }
      // KaTeX: o MathML fica em <span class="katex-mathml">.
      const kmm = el.querySelector('.katex-mathml');
      if (kmm) {
        const t = mathmlToText(kmm.querySelector('math') || kmm);
        if (t.trim()) return t;
      }
      if (el.matches('math')) {
        const t = mathmlToText(el);
        if (t.trim()) return t;
      }
      return chtmlToText(el);
    };

    const blocks = Array.from(root.querySelectorAll('mjx-container, .katex-mathml'));
    for (const b of blocks) {
      // Processa cada container uma vez: o .katex-mathml aninhado dentro de um
      // mjx-container (ou vice-versa) nao pode ser serializado em separado.
      if (b.parentElement && b.parentElement.closest('mjx-container, .katex-mathml')) continue;
      b.replaceWith(document.createTextNode(serialize(b).trim() || ' '));
    }
  }

  // Tabela -> texto alinhado por coluna.
  //
  // Sem isto, o innerText do Khan entrega os valores SEM dizer a que coluna
  // pertencem: "Distancia (quilometros)Preco (reais)110186145207170222". A IA
  // recebe seis numeros soltos, nao sabe qual e distancia e qual e preco, e
  // pede ajuda ao usuario -- foi o que aconteceu no exercicio
  // interpreting-tables, numa questao que ela resolve se souber os pares.
  //
  // O separador " | " e proposital: o prompt ja usa "  [n] " para alternativas
  // e "  [n] " para campos, entao uma barra nao colide com nada.
  function renderTablesInPlace(clone) {
    for (const table of clone.querySelectorAll('table')) {
      const linhas = [];
      // O caption (quando existe) nomeia a tabela e ajuda a IA a interpretar.
      const caption = table.querySelector('caption');
      if (caption && (caption.textContent || '').trim()) {
        linhas.push(caption.textContent.trim());
      }
      for (const tr of table.querySelectorAll('tr')) {
        const celulas = [...tr.querySelectorAll('th, td')]
          .map((c) => (c.textContent || '')
            .replace(/[\u200b\u200c\u200d\u2060\ufeff]/g, '')
            .replace(/\s+/g, ' ')
            .trim());
        // Linha so de celulas vazias nao informa nada e polui o enunciado.
        if (celulas.some((c) => c)) linhas.push(celulas.join(' | '));
      }
      if (!linhas.length) continue;
      // <pre> para o textContent carregar as quebras de linha: o innerText de
      // um clone fora do documento cai para textContent, entao o "\n" real e o
      // que sobrevive.
      const bloco = document.createElement('pre');
      bloco.textContent = `\nTABELA:\n${linhas.join('\n')}\n`;
      table.replaceWith(bloco);
    }
  }

  function extractProblem(root) {
    const clone = root.cloneNode(true);
    // A matematica precisa virar texto legivel antes de qualquer remocao: e o
    // unico momento em que a arvore matematica ainda esta no clone.
    renderMathInPlace(clone);
    // Idem para as tabelas: o clone ainda tem as celulas.
    renderTablesInPlace(clone);

    // Tira as alternativas do enunciado. O bloco de escolhas ja vai para a
    // secao ALTERNATIVAS do prompt; deixar aqui duplica o texto de forma
    // colada ("...inversivel?-5<=x<=0-1<=x<=0-1<=x<=1"), que faz a IA
    // contar opcoes que nao existem e escolher a letra errada.
    for (const g of clone.querySelectorAll('fieldset, [role="group"], [role="radiogroup"]')) {
      const items = g.querySelectorAll('li[role="listitem"]');
      if (items.length > 1) g.remove();
    }

    // Remove gabarito, gabarito parcial, dicas, controles de navegacao e o
    // rodape de "conteudo relacionado" (que traz a miniatura do video e joga
    // texto como "Video 2 minutos" dentro do enunciado).
    clone.querySelectorAll([
      'script', 'style', 'noscript', 'button', '[role="button"]',
      '.solution', '.hints', '.hints-and-solutions', '.answer_percents',
      '.response-area', '.input-area', '.sr-only', '[aria-hidden="true"]',
      '[data-testid="answer-status"]', '.feedback', '.ka-button',
      '.ka-icon', '.kaicon',
    ].join(',')).forEach((n) => n.remove());
    clone.querySelectorAll(OFF_TOPIC).forEach((n) => n.remove());

    const text = (clone.innerText || clone.textContent || '')
      // O MathJax do Khan deixa espacos de largura zero entre os simbolos
      // ("-5 <= x <= 0"); sem isso as alternativas viram "−5≤x≤0" coladas.
      .replace(/[-‏⁠﻿]/g, '')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    return stripRelatedFooter(text).slice(0, state.cfg.maxProblemChars);
  }

  // Alternativas de multipla escolha, em ordem de leitura.
  // Nos os elementos clicaveis de cada alternativa, na ordem em que aparecem.
  //
  // O padrao que realmente importa hoje e o do widget de radio do Perseus,
  // verificado no DOM real: fieldset > [role=group] > ul[role=list] >
  // li[role=listitem] > button(letra "A"). NAO existe input[type=radio] nem
  // label nessa questao -- o botao da letra e o que marca a selecao. As classes
  // ("perseus_ejJezW-c") sao hashes de CSS-in-JS e mudam a cada build, entao
  // ancoramos nos papeis ARIA, que sao estaveis.
  //
  // Os padroes antigos (data-testid, label com input) ficam como reserva para
  // outros formatos de exercicio.
  function choiceNodes(root) {
    const visible = (n) => isVisible(n) || n.getClientRects().length > 0;

    const pickers = [
      () => root.querySelectorAll('[data-testid^="answer-choice-"]'),
      () => root.querySelectorAll('label.answer-persisted'),
      () => root.querySelectorAll('.choices label'),
      () => Array.from(root.querySelectorAll('label')).filter((l) =>
        l.querySelector('input[type="radio"], input[type="checkbox"]')
      ),
      () => root.querySelectorAll('.choice'),
      // Radio do Perseus: um li por alternativa, dentro de um group/fieldset.
      () => {
        const groups = root.querySelectorAll('fieldset, [role="group"], [role="radiogroup"]');
        for (const g of groups) {
          const items = Array.from(g.querySelectorAll('li[role="listitem"]'));
          if (items.length > 1) return items;
        }
        return [];
      },
      // Radio do Perseus sem o papel de listitem: cai para os botoes-folha
      // que contem SO a letra da alternativa.
      () => {
        const groups = root.querySelectorAll('fieldset, [role="group"], [role="radiogroup"]');
        for (const g of groups) {
          const btns = Array.from(g.querySelectorAll('button')).filter((b) =>
            /^[A-Za-z]$/.test((b.innerText || '').trim())
          );
          if (btns.length > 1) return btns;
        }
        return [];
      },
    ];

    for (const pick of pickers) {
      const nodes = Array.from(pick()).filter(visible);
      if (nodes.length) return nodes;
    }
    return [];
  }

  // Texto legivel de uma alternativa, sem a letra "A"/"B" que o widget mostra
  // em um botao separado (a IA precisa ler o enunciado, nao "A f e inversivel").
  function choiceText(node) {
    const clone = node.cloneNode(true);
    // Mesma razao do enunciado: alternativa com fracao lida como texto vira
    // "9m" e a comparacao entre opcoes fica sem sentido.
    renderMathInPlace(clone);
    for (const b of clone.querySelectorAll('button')) {
      if (/^[A-Za-z]$/.test((b.innerText || '').trim())) b.remove();
    }
    // O MathJax do Khan deixa espacos de largura zero entre os simbolos
    // ("-5 <= x <= 0"). Sem a limpeza, a alternativa chega na IA como
    // "-5<=x<=0" com lixo invisivel grudado, e comparar opcoes fica dificil.
    return (clone.innerText || clone.textContent || '')
      .replace(/[‌‍‎‏﻿]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function extractChoices(root) {
    const nodes = choiceNodes(root);
    if (!nodes.length) return [];
    return nodes.slice(0, state.cfg.maxChoices).map((n, i) => ({
      index: i,
      text: choiceText(n),
      // Alternativa que e SO imagem chega com text vazio: sem esta marca a IA
      // recebe "[0] " e "[1] " e nao tem como escolher. Ver
      // capturarAlternativasSemTexto.
      temImagem: alternativaTemFigura(n),
    }));
  }

  // Ha figura NA alternativa? Aceita <img> com src e <svg> de tamanho util --
  // o segundo cobre figura desenhada no DOM, que o Khan usa em alguns widgets.
  function alternativaTemFigura(node) {
    if ([...node.querySelectorAll('img')].some((e) => (e.getAttribute('src') || '').trim())) return true;
    return [...node.querySelectorAll('svg')].some((s) => {
      const r = s.getBoundingClientRect();
      return r.width * r.height >= 40 * 30;
    });
  }

  // Campos de resposta por teclado, com o TIPO de cada um.
  //
  // Dois bugs reais encontrados no Khan de verdade:
  //  1) o `input[type=text]` que aparecia primeiro na pagina era do banner de
  //     cookies (OneTrust) -> por isso tudo e filtrado por estar dentro da raiz;
  //  2) o MathQuill NAO e um input de verdade: e um <span> editavel apoiado
  //     num <textarea> invisivel de 0x0. Nao da para preencher por `value`
  //     (ele ignora) nem por KeyboardEvent sintetico (exige evento *trusted*).
  //     Entao ele vira um tipo proprio, tratado via teclado matematico na tela.
  function extractInputs(root) {
    const inputs = [];

    // (a) Campos de matematica (MathQuill)
    for (const el of root.querySelectorAll(SEL.mathField)) {
      if (el.closest('.solution, .hints, .hints-and-solutions')) continue;
      const rendered = el.querySelector(SEL.mathRendered);
      inputs.push({
        index: inputs.length,
        kind: 'math',
        el,
        visible: isVisible(el),
        value: (rendered?.textContent || '').replace(/[\u200b\u200c\u200d]/g, '').trim(),
        placeholder: '',
        ariaLabel: el.getAttribute('aria-label') || '',
      });
      if (inputs.length >= 8) break;
    }

    // (b) Campos de texto / numero / area de texto de verdade
    if (inputs.length < 8) {
      const nodes = root.querySelectorAll(
        'input[type="text"], input[type="number"], input[type="tel"], input:not([type]), textarea'
      );
      for (const el of nodes) {
        if (el.disabled || el.readOnly) continue;
        if (el.type === 'hidden') continue;
        if (el.closest('.solution, .hints, .hints-and-solutions')) continue;
        // O textarea do MathQuill e interno: nao e um campo de texto real.
        if (el.closest(SEL.mathField) || el.closest('.mq-textarea')) continue;
        inputs.push({
          index: inputs.length,
          kind: 'text',
          el,
          visible: isVisible(el),
          value: el.value ?? '',
          placeholder: el.placeholder || '',
          ariaLabel: el.getAttribute('aria-label') || '',
        });
        if (inputs.length >= 8) break;
      }
    }

    return inputs;
  }

  // "Vamos lá" / "Começar" / "Iniciar prática": a tela de boas-vindas da unidade.
  function findStartButton() {
    const re = /^(vamos lá|vamos la|começar|comecar|iniciar prática|iniciar pratica|começar prática|praticar|start|let's go|got it)$/i;
    return (
      Array.from(document.querySelectorAll('button, [role="button"]'))
        .filter((b) => isVisible(b) && re.test((b.innerText || '').trim()))
        // o botao de comecar e sempre o maior da tela
        .sort((a, b) => b.getBoundingClientRect().width - a.getBoundingClientRect().width)[0] || null
    );
  }

  function findButtons(root) {
    // Os botoes ficam FORA de [data-assessment-item-id], na barra do exercicio.
    // Verificamos primeiro dentro da raiz e depois no documento.
    const inRoot = (btn) => btn && root && root.contains(btn);
    const doc = document;
    const byText = (re) =>
      Array.from(doc.querySelectorAll('button, [role="button"]'))
        .filter((b) => isVisible(b) && re.test((b.innerText || '').trim()))
        .sort((a, b) => (inRoot(a) ? 0 : 1) - (inRoot(b) ? 0 : 1))[0] || null;
    const byTestId = (sel) =>
      Array.from(doc.querySelectorAll(sel)).find((b) => isVisible(b)) || null;

    return {
      check:
        doc.querySelector(SEL.checkBtn) ||
        byText(/^(check|check answer|verificar|confirmar|tentar novamente)$/i),
      // O "next" do exercicio e o data-testid EXATO. Um seletor por substring
      // ("[data-testid*=next]") casa tambem com "course-next-nav", que e a seta
      // de navegacao do curso no topo da pagina: clicar nela nao troca de
      // questao, o agente fica pedindo Next para sempre e so a trava global
      // encerra o loop. Ordem importa: teste exato antes de qualquer substring.
      next:
        byTestId('[data-testid="exercise-next-question"]') ||
        byTestId('[data-testid="exercise-next-button"]') ||
        // subdivided-exercise: o proximo passo e um item da lista de subtópicos
        byTestId('[data-testid="continue-button"]') ||
        doc.querySelector(SEL.nextBtn) ||
        byText(/^(next|next question|continuar|pr[oó]xima|finalizar|concluir|continuar pr[oó]xima)$/i),
      skip: byTestId('[data-testid="exercise-skip-button"]') || doc.querySelector(SEL.skipBtn),
      hint: byText(/^(dicas?|hint|get hint|pedir dica|mostrar dica)$/i),
      startOver: byTestId('[data-testid="start-over-button"]') || doc.querySelector(SEL.startOver),
    };
  }

  // Estado da questao atual, lido dos data-testid de progresso.
  //   exercise-progress-0-incomplete-current  -> ainda nao respondida
  //   exercise-progress-0-incorrect-current   -> errada
  //   exercise-progress-0-correct-current     -> correta
  // Esse e o sinal de feedback mais confiavel que existe no Khan hoje.
  function readProgress() {
    const nodes = Array.from(document.querySelectorAll(SEL.progress));
    if (!nodes.length) return null;
    const current = nodes.find((n) => /-current$/.test(n.getAttribute('data-testid') || ''));
    const target = current || nodes[0];
    const id = target.getAttribute('data-testid') || '';
    const state = /-(correct|incorrect|incomplete)/.exec(id)?.[1] || null;
    return {
      state,
      total: nodes.length,
      done: nodes.filter((n) => /-correct/.test(n.getAttribute('data-testid') || '')).length,
      index: Number(/-progress-(\d+)-/.exec(id)?.[1] ?? 0),
    };
  }

  // Texto de feedback quando o Khan escreve uma mensagem de erro/sucesso.
  function extractFeedback(root) {
    const prog = readProgress();
    if (prog?.state === 'correct') return 'Correct!';
    if (prog?.state === 'incorrect') {
      // O Khan escreve a explicacao em texto solto perto do botao.
      const t = (document.body.innerText || '');
      const m = t.match(/(N[aã]o conseguimos entender sua resposta[^\n]*|Voc[eê] est[aá] quase l[aá]|Isso n[aã]o est[aá] correto[^\n]*|That's not right[^\n]*)/i);
      return m ? m[0].trim().slice(0, 160) : 'Incorrect.';
    }
    return null;
  }

  // Identidade ESTAVEL da questao, para a memoria do agente.
  //
  // Nao pode ser a fingerprint: ela inclui feedback e estado do progresso de
  // proposito (e a trava de travamento depende disso), entao MUDA quando o
  // aluno responde e erra. Ancorar a memoria nela fazia o agente esquecer
  // exatamente no momento em que mais precisa lembrar -- logo depois de errar,
  // que e quando a correcao importa.
  function chaveDaQuestao(snap) {
    return [
      hash(snap.problem || ''),
      (snap.choices || []).map((c) => c.text).join('|'),
      (snap.inputs || []).length,
    ].join(':');
  }

  function takeSnapshot() {
    const root = findExerciseRoot();
    if (!root) return null;

    const problem = extractProblem(root);
    if (!problem || problem.length < 3) return null;

    const choices = extractChoices(root);
    const inputs = extractInputs(root);
    const buttons = findButtons(root);
    // O fingerprint descreve "qual exercicio eu estou olhando". Progresso e
    // feedback entram de proposito: sem eles, responder certo nao muda nada no
    // fingerprint e a trava de travamento (abaixo) acha que a pagina travou
    // justo quando ela acaba de avanchar.
    const progress = readProgress();
    const feedback = extractFeedback(root);

    const fingerprint = [
      hash(problem),
      choices.map((c) => c.text).join('|'),
      inputs.length,
      progress ? `${progress.done}/${progress.total}/${progress.state}` : '-',
      feedback || '-',
    ].join(':');
    const attempts = state.seen.get(fingerprint) || 0;

    return {
      root, problem, choices, inputs, buttons, feedback, progress,
      fingerprint, attempts,
      visual: looksVisual(root, problem),
      hasCheck: !!buttons.check,
      hasNext: !!buttons.next,
    };
  }

  // Questao com enunciado mas SEM nenhum controle (nem alternativa, nem campo).
  //
  // E o sintoma que mais custou a diagnosticar: "0 alt | 0 campos" numa pagina
  // que TEM como responder. Aconteceu porque o Khan troca o renderizador no
  // celular e o widget nao era reconhecido. Em vez de a IA pedir ajuda as
  // cegas, o script diz o que encontrou -- foi este relatorio que revelou o
  // `.keypad-input`.
  //
  // So loga uma vez por questao: o loop observa a cada 1,5 s e isto viraria
  // spam enquanto o estado persistisse.
  function diagnosticarSemControles(snap) {
    const chave = chaveDaQuestao(snap);
    if (state.diagChave === chave) return;
    state.diagChave = chave;

    const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
    const root = snap.root;

    const controles = [...document.querySelectorAll('input, textarea, [contenteditable], [role="textbox"]')]
      .filter((e) => !/ka-|khan-agent/.test(String(e.className || '')) && vis(e))
      .map((e) => ({ e, naRaiz: !!(root && root.contains(e)) }));

    const widgets = root
      ? [...root.querySelectorAll('[role="textbox"], [class*="input"], [class*="keypad"], [class*="widget"]')]
        .filter(vis).slice(0, 6)
      : [];

    log('Questao SEM alternativa e SEM campo detectado. O que a pagina tem:', 'warn');
    log(`  controles visiveis fora da raiz: ${controles.filter((c) => !c.naRaiz).length}`, 'warn');
    for (const c of controles.filter((x) => !x.naRaiz).slice(0, 4)) {
      log(`    ${c.e.tagName}.${String(c.e.className || '').slice(0, 46)}`, 'warn');
    }
    log(`  widgets dentro da raiz: ${widgets.length}`, 'warn');
    for (const w of widgets) {
      const r = w.getBoundingClientRect();
      log(`    ${w.tagName}.${String(w.className || '').slice(0, 46)} ${Math.round(r.width)}x${Math.round(r.height)}`, 'warn');
    }
    log('  (se houver um widget de resposta na lista, o seletor dele precisa entrar em SEL)', 'warn');
  }

  // Relatorio do que o script enxerga na pagina. Serve para o botao "Inspecionar"
  // e para o log quando o agente nao acha exercicio nenhum: em vez de falhar em
  // silencio, mostra exatamente quais seletores casaram e quais nao.
  function inspectPage() {
    const root = findExerciseRoot();
    const count = (sel) => document.querySelectorAll(sel).length;
    const lines = [
      `URL: ${location.href}`,
      `Raiz do exercicio: ${
        root ? `<${root.tagName.toLowerCase()}>` + (root.className ? `.${String(root.className).split(/\s+/)[0]}` : '') : 'NAO ENCONTRADA'
      }`,
      '',
      'Seletores de raiz:',
      ...EXERCISE_SELECTORS.map((s) => `  ${count(s)}x  ${s}`),
      '',
      'Controles:',
      `  ${count(SEL.checkBtn)}x  ${SEL.checkBtn}`,
      `  ${count(SEL.skipBtn)}x  ${SEL.skipBtn}`,
      `  ${count(SEL.progress)}x  ${SEL.progress}`,
      `  ${count(SEL.mathField)}x  ${SEL.mathField}  (MathQuill)`,
      `  ${count('input[type=radio]')}x  input[type=radio]`,
      `  ${count('input[type=text], input[type=number], textarea')}x  campos de texto`,
      '',
    ];
    if (!root) {
      lines.push('Nada casou. O Khan pode ter mudado o DOM, ou a pagina nao e um exercicio.');
      return lines.join('\n');
    }
    const snap = takeSnapshot();
    lines.push(
      `Enunciado: ${snap ? snap.problem.slice(0, 220) : '(nao extraido)'}`,
      `Alternativas: ${snap ? snap.choices.length : 0}`,
      `Campos: ${snap ? snap.inputs.length : 0}${
        snap && snap.inputs.length
          ? ' (' + snap.inputs.map((i) => `${i.kind}${i.value ? `="${i.value}"` : ''}`).join(', ') + ')'
          : ''
      }`,
      `Verificar: ${snap?.hasCheck ? 'sim' : 'NAO'}   Avancar: ${snap?.hasNext ? 'sim' : 'nao'}`,
      `Progresso: ${snap?.progress ? `${snap.progress.done}/${snap.progress.total} (${snap.progress.state})` : 'n/d'}`
    );
    return lines.join('\n');
  }

  /* ===========================================================================
   * EXECUCAO DAS ACOES (a parte que "age como o usuario")
   * ======================================================================== */

  // Clique com a sequencia de eventos que o React/jQuery do Khan espera.
  //
  // `at` escolhe o ponto do clique. O padrao e o CENTRO do elemento, e isso e
  // obrigatorio, nao cosmetico: o MathQuill posiciona o cursor pelo par
  // (clientX, clientY) do evento. Sem coordenadas o valor vai 0 e o clique cai
  // fora do campo -- o cursor nao anda e a tecla "Excluir" vira no-op (foi
  // exatamente o sintoma de "o agente nao consegue limpar a resposta").
  // Cria um evento de UI sem quebrar no sandbox do gerenciador.
  //
  // O Firefox exige que `view` seja uma Window de VERDADE e recusa o resto:
  //   "PointerEvent constructor: 'view' member of UIEventInit does not
  //    implement interface Window."
  // Dentro do sandbox do Tampermonkey, o `window` que o script enxerga nao
  // passa nesse teste. Entao `view: window` derrubava o loop inteiro no Firefox
  // Android: o clique nunca acontecia, e o erro subia como "Erro no loop" sem
  // dizer que a causa era o construtor do evento -- o sintoma parecia um
  // problema do Khan, e nao do clique.
  //
  // A tentativa COM `view` fica porque no Chrome ele ajuda a emular um clique
  // real. Se o navegador recusar, o mesmo evento e criado sem ele, o que e
  // perfeitamente valido: nenhum handler do Khan depende de `event.view` para
  // um clique sintetico.
  function criarEventoUI(Ctor, tipo, opts) {
    if (typeof Ctor !== 'function') return null;
    try {
      return new Ctor(tipo, { ...opts, view: window });
    } catch {
      try { return new Ctor(tipo, opts); } catch { return null; }
    }
  }

  function humanClick(el, at) {
    if (!el) return false;
    if (!isVisible(el)) {
      const fallback = el.closest('label') || el.parentElement;
      if (fallback && fallback !== el) return humanClick(fallback);
      return false;
    }
    const r = el.getBoundingClientRect();
    const opts = {
      bubbles: true,
      cancelable: true,
      button: 0,
      clientX: r.left + r.width / 2,
      clientY: r.top + r.height / 2,
      ...(at || {}),
    };
    try { el.focus({ preventScroll: true }); } catch { /* noop */ }

    const disparar = (Ctor, tipo, extra) => {
      const ev = criarEventoUI(Ctor, tipo, extra ? { ...opts, ...extra } : opts);
      if (ev) el.dispatchEvent(ev);
    };
    disparar(PointerEvent, 'pointerdown', { pointerType: 'mouse' });
    disparar(MouseEvent, 'mousedown');
    disparar(PointerEvent, 'pointerup', { pointerType: 'mouse' });
    disparar(MouseEvent, 'mouseup');
    disparar(MouseEvent, 'click');
    return true;
  }

  // Escreve num input[field] de forma que o React perceba a mudanca.
  function setInputValue(el, value) {
    const proto = Object.getPrototypeOf(el);
    const desc = Object.getOwnPropertyDescriptor(proto, 'value');
    if (desc && desc.set) desc.set.call(el, value);
    else el.value = value;

    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    if (typeof el.blur === 'function') el.blur();
  }

  // LaTeX -> notacao linear que o parser do Khan entende ("1/2", "x^2", "3^(1/2)").
  function latexToLinear(latex) {
    let s = String(latex || '').trim();

    // \frac{a}{b} e \dfrac/\tfrac -> a/b (ou (a)/(b) se o pedaco exigir par deenteses)
    for (let i = 0; i < 6; i++) {
      const m = s.match(/\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/);
      if (!m) break;
      s = s.replace(m[0], `${wrapIfComposite(m[1])}/${wrapIfComposite(m[2])}`);
    }

    function wrapIfComposite(inner) {
      const v = latexToLinear(inner);
      return /[+\-*/^]/.test(v) ? `(${v})` : v;
    }

    s = s.replace(/\\sqrt\s*\{([^{}]*)\}/g, (_m, a) => `${wrapIfComposite(a)}^(1/2)`);
    s = s.replace(/\\(?:cdot|times)/g, '*');
    s = s.replace(/\\div/g, '/');
    s = s.replace(/\\pm/g, '+-');
    s = s.replace(/\\pi/g, 'pi');
    s = s.replace(/\\theta/g, 'theta');
    s = s.replace(/\\(?:left|right|,|;|!|quad|qquad)/g, '');
    s = s.replace(/\^\{([^{}]*)\}/g, '^($1)');
    s = s.replace(/_\{([^{}]*)\}/g, '_($1)');
    s = s.replace(/\\text\s*\{([^{}]*)\}/g, '$1');
    s = s.replace(/\\mathrm\s*\{([^{}]*)\}/g, '$1');
    s = s.replace(/\\[a-zA-Z]+/g, '');   // comando residual
    s = s.replace(/\s+/g, '').trim();
    return s;
  }

  /* ===========================================================================
   * TECLADO MATEMATICO (MathQuill) -- o unico caminho viavel
   * ========================================================================
   *
   * Um campo de matematica do Khan NAO e um <input>. E um <span class="mq-*">
   * editavel apoiado num <textarea> invisivel de 0x0. Verificado no DOM real:
   *
   *   - atribuir .value no textarea .......... IGNORADO
   *   - document.execCommand('insertText') ... IGNORADO (retorna true, nao faz nada)
   *   - KeyboardEvent sintetico .............. IGNORADO (MathQuill exige evento *trusted*,
   *                                            que nenhum userscript consegue gerar)
   *   - Clicar nos botoes do teclado da tela . FUNCIONA
   *
   * Duas regras que makes a digitacao possivel e quecustam uma tarde para achar:
   *
   *   1) "Fração, sem a expressão atual" insere uma fracao VAZIA no cursor e
   *      coloca o cursor dentro do NUMERADOR (ele nao involve o que ja estava).
   *   2) Clicar em .mq-denominator MOXE o cursor para o denominador.
   *      Sem isso nao ha como digitar o denominador (a seta para baixo seria
   *      um evento de teclado *trusted*).
   *
   * Vocabulario real das teclas (aria-label), por aba:
   *   Numbers   : 0-9, Decimal, Negativo, Porcentagem
   *   Extras    : x
   *   Operators : Quadrado, Personalizar expoente, Subscrito,
   *               Raiz quadrada, Radical com raiz personalizada
   *   (comuns a todas) : Fração..., Mais, Menos, Multiplicar,
   *                      Parêntese esquerdo/direito, Excluir
   *
   * A unica letra disponivel e "x". Qualquer outro simbolo e reportado de volta
   * para a IA como "nao disponivel", em vez de falhar calado.
   */

  const MATH_TABS = ['Numbers', 'Extras', 'Operators'];

  // Aba preferida de cada tecla, so para reduzir trocas de aba.
  const MATH_TAB_OF = {
    Decimal: 'Numbers', Negativo: 'Numbers', Porcentagem: 'Numbers',
    'Raiz quadrada': 'Operators', 'Radical com raiz personalizada': 'Operators',
    Quadrado: 'Operators', 'Personalizar expoente': 'Operators', Subscrito: 'Operators',
  };

  function findMathKeyBtn(label) {
    const want = String(label).trim().toLowerCase();
    return (
      Array.from(document.querySelectorAll('button[aria-label]')).find((b) => {
        if (!isVisible(b)) return false;
        const l = (b.getAttribute('aria-label') || '').trim().toLowerCase();
        // "Fração, sem a expressão atual" tambem casa com "Fração"
        return l === want || l.startsWith(want + ',');
      }) || null
    );
  }

  // As teclas do teclado matematico NAO podem usar humanClick: aquele chama
  // focus() antes do clique, e o focus() no botao rouba o foco do MathQuill e
  // perde o cursor -- a tecla "clica" e nada acontece. Com click() puro o
  // React responde e o MathQuill mantem a selecao (verificado no DOM real).
  function clickKeyBtn(btn) {
    if (!btn) return false;
    try { btn.click(); return true; } catch { return humanClick(btn); }
  }

  function clickMathTab(tab) {
    return clickKeyBtn(findMathKeyBtn(tab));
  }

  // O teclado matematico FECHA SOZINHO em varios momentos (ao conferir a
  // resposta, ao re-renderizar o React, ao perder o foco). Depois disso TODA
  // digitacao vira no-op silencioso -- foi exatamente o que aconteceu no
  // primeiro teste end-to-end, com o agente "digitando" nada. Por isso cada
  // tecla verifica se o teclado ainda esta na tela e reabre se preciso.
  function mathKeyboardIsOpen() {
    return !!findMathKeyBtn('Excluir');
  }

  async function ensureMathKeyboard() {
    if (mathKeyboardIsOpen()) return true;
    const btn = document.querySelector(SEL.mathKeyboardOpen);
    if (!btn || !isVisible(btn)) return false;
    clickKeyBtn(btn);
    await humanDelay(200);
    return mathKeyboardIsOpen();
  }

  // Clica numa tecla, reabrindo/trocando de aba se necessario.
  async function mathPress(label) {
    const tryClick = () => clickKeyBtn(findMathKeyBtn(label));

    if (await ensureMathKeyboard() && tryClick()) return true;

    const preferred = MATH_TAB_OF[label] || (label === 'x' ? 'Extras' : null);
    const order = preferred
      ? [preferred, ...MATH_TABS.filter((t) => t !== preferred)]
      : MATH_TABS.slice();
    for (const tab of order) {
      if (!clickMathTab(tab)) continue;
      await humanDelay(120);
      if (tryClick()) return true;
    }
    return false;
  }

  function openMathKeyboard() {
    if (mathKeyboardIsOpen()) return true;
    const btn = document.querySelector(SEL.mathKeyboardOpen);
    if (!btn || !isVisible(btn)) return false;
    return clickKeyBtn(btn);
  }

  function closeMathKeyboard() {
    const b = document.querySelector(SEL.mathKeyboardClose) || findMathKeyBtn('fechar teclado matemático');
    if (b) humanClick(b);
  }

  // Le o que o MathQuill renderizou.
  //
  // NAO da para usar textContent: o MathQuill desenha a fracao com 21 em cima
  // de 5, entao o textContent sai "215" e nao "21/5" -- o log mentia e a IA
  // via a resposta errada. Alem disso parentes, raiz e radical sao desenhados
  // com CSS, sem caractere nenhum no DOM. Entao percorremos a arvore.
  function readMathField(field) {
    const root = field.querySelector(SEL.mathRendered);
    if (!root) return '';
    return mqNormalize(mqSerialize(root));
  }

  // As classes do MathQuill precisam ser comparadas por TOKEN, nunca por
  // substring. Os containers se chamam "mq-bracket-container" e
  // "mq-sqrt-container", que contem "mq-bracket"/"mq-sqrt" como substring --
  // com includes() o branch disparava no container, emitia "["/"sqrt(" e
  // `continue` sem percorrer os filhos, entao o miolo sumia inteiro
  // (o log dizia "(4+2)*(3-1) -> [*[").
  function mqHas(node, token) {
    const c = node && typeof node.className === 'string' ? node.className : '';
    return c.split(/\s+/).includes(token);
  }

  // Um numerador/argumento so precisa de parenteses se tiver soma, subtracao
  // interna, multiplicacao, divisao ou potencia. O sinal de menos da FRENTE
  // nao conta: "-3" e um "-3" legitimo, nao precisa virar "(-3)".
  function mqNeedsParens(s) {
    const t = s.startsWith('-') ? s.slice(1) : s;
    return /[+\-*/^]/.test(t);
  }

  function mqSerialize(el) {
    if (!el) return '';
    let out = '';
    for (const node of el.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) { out += node.textContent; continue; }
      if (node.nodeType !== Node.ELEMENT_NODE) continue;
      out += mqSerializeNode(node);
    }
    return out;
  }

  function mqSerializeNode(node) {
    if (node.tagName === 'svg' || node.tagName === 'path' || mqHas(node, 'mq-cursor')) return '';

    // Fracao: numerador sobre denominador. Monta "a/b" de verdade.
    if (mqHas(node, 'mq-fraction')) {
      const n = mqSerialize(node.querySelector(':scope > .mq-numerator'));
      const d = mqSerialize(node.querySelector(':scope > .mq-denominator'));
      return `${mqNeedsParens(n) ? `(${n})` : n}/${d || '?'}`;
    }

    // Raiz quadrada: container -> prefixo(svg) + haste(radicando). O radicando
    // esta em .mq-sqrt-stem, e o prefixo e so o desenho do simbolo.
    if (mqHas(node, 'mq-sqrt-container')) {
      const stem = node.querySelector(':scope > .mq-sqrt-stem');
      return `sqrt(${mqSerialize(stem)})`;
    }
    // Bloco de raiz ("Raiz quadrada" com radicando ja montado).
    if (mqHas(node, 'mq-sqrt-block')) {
      return `sqrt(${mqSerialize(node.querySelector(':scope > .mq-sqrt-stem') || node)})`;
    }
    if (mqHas(node, 'mq-root-symbol')) return '';

    // Parenteses: desenhados em SVG, sem caractere no DOM. O lado vem de
    // "mq-bracket-l"/"mq-bracket-r" (e nao de "mq-left"/"mq-right", que o
    // MathQuill nao usa nessa estrutura).
    if (mqHas(node, 'mq-bracket-l')) return '(';
    if (mqHas(node, 'mq-bracket-r')) return ')';
    if (mqHas(node, 'mq-paren')) return mqHas(node, 'mq-right') ? ')' : '(';
    if (mqHas(node, 'mq-bracket')) return mqHas(node, 'mq-right') ? ']' : '[';

    // Expoente/subescrito. A BASE fica FORA do .mq-supsub (e o irmao
    // anterior), por isso aqui so se le o sup/sub.
    if (mqHas(node, 'mq-supsub')) {
      const sup = node.querySelector(':scope > .mq-sup');
      const sub = node.querySelector(':scope > .mq-sub');
      let out = '';
      if (sup) out += '^(' + mqSerialize(sup) + ')';
      if (sub) out += '_(' + mqSerialize(sub) + ')';
      return out;
    }
    if (mqHas(node, 'mq-exponent')) return '^(' + mqSerialize(node.querySelector(':scope > .mq-exponent')) + ')';
    if (mqHas(node, 'mq-subscript')) return '_(' + mqSerialize(node.querySelector(':scope > .mq-subscript')) + ')';

    return mqSerialize(node);
  }

  // Limpa os simbolos que o MathQuill desenha com fonte tipografica e
  // normaliza para notacao linear, para a IA ler "4x - 156x" e nao "4x 156x".
  function mqNormalize(s) {
    return s
      .replace(/[\u200b\u200c\u200d\u200e\u200f\u00a0]/g, '')
      .replace(/[\u00d7\u00b7\u22c5\u2219\u2217\u22c6]/g, '*')
      .replace(/\u2212/g, '-')
      .replace(/\u00f7/g, '/')
      .replace(/\u2260/g, '!=')
      .replace(/\u2264/g, '<=')
      .replace(/\u2265/g, '>=')
      .replace(/\s+/g, '')
      .trim();
  }

  // Coloca o cursor no FIM do campo.
  //
  // Importante demais: clicar no CENTRO do campo pode cair no meio da
  // expressao, e ai o "Excluir" (que apaga para a esquerda) so apaga a metade
  // da esquerda e o campo nunca fica vazio -- a resposta antiga se acumula com
  // a nova. Clicar na BORDA DIREITA do bloco renderizado resolve, porque o
  // hit-test do MathQuill cai depois do ultimo simbolo.
  function focusMathFieldEnd(field) {
    const root = field.querySelector(SEL.mathRendered);
    if (!root) return humanClick(field);
    for (const fx of [0.995, 1, 0.97, 0.9]) {
      const r = root.getBoundingClientRect();
      if (r.width < 2) break;
      humanClick(root, { clientX: r.left + r.width * fx, clientY: r.top + r.height / 2 });
      const cur = root.querySelector('.mq-hasCursor');
      if (!cur || !cur.closest(NESTED)) return true;
    }
    return false;
  }

  // Limpa o campo. Precisa do teclado aberto: "Excluir" e uma tecla dele.
  // Devolve um diagnostico em vez de so true/false, porque "nao limpou" e um
  // defeito silencioso que so aparece como expressao duplicada no log.
  async function clearMathField(field) {
    // O teste de "vazio" usa a serializacao, nao a classe .mq-empty do CSS:
    // depois de apagar, o MathQuax sometimes deixa o bloco sem .mq-empty e so
    // com o cursor, e a classe dava falso negativo (o campo ja estava limpo,
    // mas o script achava que nao conseguiu limpar e reclamava).
    const empty = () => readMathField(field) === '';
    if (empty()) return { ok: true, motivo: 'ja estava vazio' };

    focusMathFieldEnd(field);
    if (!(await ensureMathKeyboard())) return { ok: false, motivo: 'teclado nao abriu' };
    focusMathFieldEnd(field);

    let teclas = 0;
    for (let i = 0; i < 120; i++) {
      if (empty()) break;
      if (!clickKeyBtn(findMathKeyBtn('Excluir'))) return { ok: false, motivo: `tecla Excluir sumiu apos ${teclas}` };
      teclas++;
    }
    return empty()
      ? { ok: true, motivo: `${teclas} tecla(s)` }
      : { ok: false, motivo: `apos ${teclas} tecla(s) ficou "${readMathField(field).slice(0, 30)}"` };
  }

  /* ===========================================================================
   * DIGITACAO NO MATHQUILL DO CELULAR (eventos sinteticos)
   * ---------------------------------------------------------------------------
   * O Khan troca o renderizador da resposta no celular: em vez de um
   * <input type=text> (desktop), usa um <div class="keypad-input"> com um
   * .mq-root-block do MathQuill dentro. E nao desenha teclado na tela nenhum --
   * quem digita e o teclado do sistema, que o agente nao tem.
   *
   * Isto foi medido, nao suposto. O MathQuill do mobile ACEITA eventos
   * sinteticos, com duas condicoes que custaram a achar:
   *
   *   1. Os TRES eventos por caractere: keydown + keypress + keyup. So keydown
   *      nao entra, so keypress nao entra, e keydown sem keyCode tambem nao.
   *   2. Uma PAUSA entre caracteres. Sem pausa o MathQuill reposiciona o cursor
   *      de forma assincrona e so o ULTIMO caractere sobrevive -- "123" virava
   *      "3". Com 20 ms ja funciona; usamos 25 ms de folga.
   *
   * O que NAO funciona (testado): execCommand('insertText'), InputEvent
   * sintetico, e escrever no value do textarea escondido. Por isso o caminho
   * nao e "digitar texto", e "emitir teclas".
   *
   * Limite conhecido: '^' nao produz expoente por este caminho (o MathQuill
   * espera o comando, nao o caractere). O `warn` da resposta diz quando o campo
   * ficou diferente do pedido, em vez de mentir que deu certo.
   * ======================================================================== */
  async function digitarPorEventos(field, texto, { pausaMs = 25 } = {}) {
    // No destacado nao recebe nada: o React pode substituir o campo no meio do
    // caminho, e o sintoma e o campo ficar vazio sem erro. Devolver '' faz o
    // chamador reconsultar o DOM vivo e tentar de novo.
    if (!field || !field.isConnected) return '';
    field.focus();
    for (const ch of String(texto)) {
      if (!field.isConnected) return '';
      const cc = ch.charCodeAt(0);
      const opcoes = {
        key: ch, keyCode: cc, which: cc, charCode: cc,
        bubbles: true, cancelable: true, composed: true,
      };
      field.dispatchEvent(new KeyboardEvent('keydown', opcoes));
      field.dispatchEvent(new KeyboardEvent('keypress', opcoes));
      field.dispatchEvent(new KeyboardEvent('keyup', opcoes));
      await sleep(pausaMs);
    }
    return readMathField(field);
  }

  async function apagarPorEventos(field, max = 80) {
    if (!field || !field.isConnected) return '';
    field.focus();
    for (let i = 0; i < max; i++) {
      if (!field.isConnected) return '';
      if (readMathField(field) === '') break;
      const opcoes = {
        key: 'Backspace', keyCode: 8, which: 8, charCode: 8,
        bubbles: true, cancelable: true, composed: true,
      };
      field.dispatchEvent(new KeyboardEvent('keydown', opcoes));
      field.dispatchEvent(new KeyboardEvent('keypress', opcoes));
      field.dispatchEvent(new KeyboardEvent('keyup', opcoes));
      await sleep(25);
    }
    return readMathField(field);
  }

  // Compara o que foi pedido com o que ficou no campo, ignorando as diferencas
  // que o proprio Khan introduz ao renderizar: o MathQuill troca "." por ",",
  // "/" por "÷" e "-" por "−".
  function mesmaExpressao(a, b) {
    const norm = (s) => String(s || '')
      .replace(/[\s\u200b\u200c\u200d\ufeff]/g, '')
      .replace(/,/g, '.')
      .replace(/÷/g, '/')
      .replace(/[−–—]/g, '-');
    return norm(a) === norm(b);
  }

  // Move o cursor para o denominador da ultima fracao.
  function focusDenominator(field) {    const den = field.querySelector(`${SEL.mathRendered} .mq-fraction .mq-denominator`);
    if (!den) return false;
    const r = den.getBoundingClientRect();
    humanClick(den, { clientX: r.x + 2, clientY: r.y + 2 });
    return !!field.querySelector(`${SEL.mathRendered} .mq-denominator.mq-hasCursor`);
  }

  // Tira o cursor de dentro de um container aninhado (fracao, expoente,
  // subescrito, raiz) e o devolve para a raiz, no fim da expressao.
  //
  // Sem isso tudo que vier depois fica preso dentro: "21 / (5+4x)" em vez de
  // "4x + 21/5", e "x^(-156x+46)" em vez de "x^2 - 156x".
  //
  // Funciona clicando no bloco renderizado na BORDA DIREITA: o hit-test do
  // MathQuill usa a coordenada X e cai depois do container. Clicar no meio do
  // bloco, ou no .mq-denominator vazio (largura 0), nao funciona.
  // NESTED precisa listar as classes EXATAS que o MathQuill usa. Um seletor
  // ".mq-sqrt" nao casa com "mq-sqrt-container" (seletor CSS casa token, nao
  // substring) -- por isso o "+2" de sqrt(16)+2 ficava preso dentro da raiz.
  const NESTED = [
    '.mq-fraction',
    '.mq-supsub',
    '.mq-sqrt-container',
    '.mq-sqrt-block',
    '.mq-bracket-container',
    '.mq-bracket-middle',
  ].join(', ');

  function exitToRootEnd(field) {
    const root = field.querySelector(SEL.mathRendered);
    if (!root) return false;
    const stuck = () => {
      const cur = root.querySelector('.mq-hasCursor');
      return !!(cur && cur.closest(NESTED));
    };
    if (!stuck()) return true;

    for (const fx of [0.97, 0.995, 1]) {
      const r = root.getBoundingClientRect();
      if (r.width < 4) break;
      humanClick(root, { clientX: r.left + r.width * fx, clientY: r.top + r.height / 2 });
      if (!stuck()) return true;
    }
    return false;
  }

  /**
   * Converte LaTeX numa lista de operacoes de teclado.
   * Retorna tambem `unsupported`: simbolos que o teclado do Khan nao oferece.
   */
  function latexToMathOps(latex) {
    const src = String(latex || '')
      .replace(/\\(?:left|right|,|;|!|:|quad|qquad|displaystyle|textstyle|thinspace|;)\s*/g, '')
      .replace(/\\[dt]?frac/g, '\\FRAC')
      .replace(/\\sqrt/g, '\\SQRT')
      .replace(/\\cdot|\\times/g, '*')
      .replace(/\\div/g, '/');

    const unsupported = [];
    let i = 0;

    const skipWs = () => { while (i < src.length && /\s/.test(src[i])) i++; };

    function parseArg() {
      skipWs();
      if (src[i] === '{') { i++; return parseSeq(); }
      const ch = src[i];
      if (ch === undefined) return [];
      i++;
      if (/[0-9]/.test(ch)) return [{ k: 'digit', v: ch }];
      if (ch === 'x') return [{ k: 'letter', v: 'x' }];
      if (ch === '-') return [{ k: 'key', label: 'Menos' }];
      unsupported.push(ch);
      return [];
    }

    function parseSeq() {
      const ops = [];
      while (i < src.length) {
        const ch = src[i];
        if (ch === '}') { i++; break; }

        if (ch === '{') { i++; ops.push(...parseSeq()); continue; }

        if (src.startsWith('\\FRAC', i)) {
          i += 5;
          const num = parseArg();
          const den = parseArg();
          ops.push({ k: 'frac', num, den });
          continue;
        }
        if (src.startsWith('\\SQRT', i)) {
          i += 5;
          ops.push({ k: 'sqrt', arg: parseArg() });
          continue;
        }
        if (ch === '^') { i++; ops.push({ k: 'pow', exp: parseArg() }); continue; }
        if (ch === '_') { i++; ops.push({ k: 'sub', arg: parseArg() }); continue; }
        if (ch === '/') {
          // "a/b" entre atomos simples vira fracao; o resto e insuportavel.
          const prev = ops.pop();
          skipWs();
          const nxt = src[i];
          if (prev && nxt) {
            const den = parseArg();
            ops.push({ k: 'frac', num: [prev], den });
          } else {
            unsupported.push('/');
          }
          continue;
        }

        const simple = { ')': 'Parêntese direito', '(': 'Parêntese esquerdo', '+': 'Mais', '-': 'Menos', '*': 'Multiplicar', '.': 'Decimal', '%': 'Porcentagem' };
        if (/[0-9]/.test(ch)) { ops.push({ k: 'digit', v: ch }); i++; continue; }
        if (ch === 'x') { ops.push({ k: 'letter', v: 'x' }); i++; continue; }
        if (simple[ch]) { ops.push({ k: 'key', label: simple[ch] }); i++; continue; }
        if (/[a-zA-Z]/.test(ch)) { unsupported.push(ch); i++; continue; }
        if (/\s/.test(ch)) { i++; continue; }
        unsupported.push(ch);
        i++;
      }
      return ops;
    }

    const ops = parseSeq();
    return { ops, unsupported: [...new Set(unsupported)] };
  }

  // Executa a lista de operacoes no MathQuill. Devolve o que foi de fato
  // digitado, para o log e para a IA reaprender em caso de erro.
  async function runMathOps(ops, field, pressed) {
    for (const op of ops) {
      switch (op.k) {
        case 'digit':
          if (await mathPress(op.v)) pressed.push(op.v);
          else pressed.push('?');
          break;
        case 'letter':
          if (await mathPress('x')) pressed.push('x');
          else pressed.push('?');
          break;
        case 'key':
          if (await mathPress(op.label)) pressed.push(op.label);
          else pressed.push('?');
          break;
        case 'frac':
          if (await mathPress('Fração')) pressed.push('frac');
          else pressed.push('?');
          await runMathOps(op.num, field, pressed);
          if (focusDenominator(field)) {
            await runMathOps(op.den, field, pressed);
            if (exitToRootEnd(field)) pressed.push('/');
            else pressed.push('(preso na fracao)');
          } else {
            pressed.push('(sem denominador)');
          }
          break;
        case 'sqrt':
          if (await mathPress('Raiz quadrada')) pressed.push('sqrt(');
          else pressed.push('?');
          await runMathOps(op.arg, field, pressed);
          if (exitToRootEnd(field)) pressed.push(')');
          break;
        case 'pow':
          if (await mathPress('Personalizar expoente')) pressed.push('^');
          else pressed.push('?');
          await runMathOps(op.exp, field, pressed);
          // Sem sair do expoente, o resto da expressao entra dentro dele.
          if (exitToRootEnd(field)) pressed.push('');
          else pressed.push('(preso no expoente)');
          break;
        case 'sub':
          if (await mathPress('Subscrito')) pressed.push('_');
          else pressed.push('?');
          await runMathOps(op.arg, field, pressed);
          if (exitToRootEnd(field)) pressed.push('');
          else pressed.push('(preso no subescrito)');
          break;
        default:
          break;
      }
    }
  }

  const ACTIONS = {
    async select_choice(params) {
      const snap = state.snapshot;
      if (!snap || !snap.choices.length) return { ok: false, reason: 'sem alternativas' };
      const i = Number(params.choice_index);
      if (!Number.isInteger(i) || i < 0 || i >= snap.choices.length) {
        return { ok: false, reason: `indice ${params.choice_index} invalido` };
      }
      const node = choiceNodes(snap.root)[i];
      if (!node) return { ok: false, reason: 'alternativa nao encontrada' };

      // Clica no alvo real de selecao, na ordem de especificidade:
      // o botao da letra (padrao do Perseus), o input, o label, e por fim o
      // proprio item da lista. Sem clicar no botao da letra o radio nunca
      // marca e o Check continua dizendo que falta responder.
      const letra = Array.from(node.querySelectorAll('button'))
        .find((b) => /^[A-Za-z]$/.test((b.innerText || '').trim()));
      const input = node.matches('label') ? node : node.querySelector('input, label');
      const alvo = letra || input || node;
      if (!humanClick(alvo)) return { ok: false, reason: 'no click' };
      if (alvo !== node) humanClick(node);
      return { ok: true, detail: `alternativa [${i}] "${snap.choices[i].text.slice(0, 40)}"` };
    },

    async type_answer(params) {
      const snap = state.snapshot;
      const specs = Array.isArray(params.inputs) ? params.inputs : [];
      if (!snap || !snap.inputs.length) return { ok: false, reason: 'sem campos' };
      if (!specs.length) return { ok: false, reason: 'params.inputs vazio' };

      let filled = 0;
      const skipped = [];
      const naoAplicou = [];

      for (const spec of specs) {
        const idx = Number(spec.index);
        const field = snap.inputs[idx];
        if (!field) continue;
        // Campo MathQuill nao aceita `value`: isso e trabalho do build_expression.
        if (field.kind === 'math') {
          skipped.push(`[${field.index}] math`);
          continue;
        }
        const text = String(spec.text ?? '');
        const valor = text.includes('\\') ? latexToLinear(text) : text;

        // Escreve e CONFIRMA no DOM vivo, reextraindo o campo.
        //
        // O elemento guardado no snapshot pode ter sido substituido pelo React
        // entre a observacao e agora -- o proprio preencher dispara re-render.
        // Escrever num no destacado nao muda nada na tela, e o log dizia
        // "1 campo(s) preenchido(s)" enquanto o usuario via o valor ANTIGO.
        //
        // A conferencia NAO pode ler o elemento em que escrevemos: se ele
        // estiver destacado, a leitura devolve o que acabamos de escrever e o
        // teste passaria mentindo. Por isso o valor e lido do campo que a
        // REEXTRACAO encontra no mesmo indice.
        let aplicou = false;
        for (let tentativa = 1; tentativa <= 3 && !aplicou; tentativa++) {
          const alvo = fieldDeTextoVivo(idx, field);
          if (!alvo) break;
          setInputValue(alvo, valor);
          humanClick(alvo);
          await humanDelay();
          // Verifica rapido: o setter nativo ja dispara input+change, e o
          // React costuma refletir em dezenas de ms.
          await sleep(60);
          aplicou = valorNoCampoVivo(idx) === valor.trim();
        }
        if (aplicou) filled++;
        else naoAplicou.push(`[${idx}]`);
      }

      if (!filled) {
        return {
          ok: false,
          reason: naoAplicou.length
            ? `o campo ${naoAplicou.join(', ')} nao aceitou o valor (a pagina pode ter re-renderizado)`
            : (skipped.length
              ? `campo(s) ${skipped.join(', ')} sao MathQuill: use build_expression com LaTeX`
              : 'nenhum indice valido'),
        };
      }
      return {
        ok: true,
        detail: `${filled} campo(s) preenchido(s)`,
        ...(skipped.length ? { warn: `ignorado: ${skipped.join(', ')}` } : {}),
        ...(naoAplicou.length ? { warn: `${naoAplicou.join(', ')} nao aceitou o valor` } : {}),
      };
    },

    async build_expression(params) {
      const snap = state.snapshot;
      if (!snap) return { ok: false, reason: 'sem snapshot' };

      const latex = String(params.latex ?? '').trim();
      if (!latex) return { ok: false, reason: 'params.latex vazio' };

      // Alvos: primeiro os campos MathQuill, depois um campo de texto comum.
      const mathField = snap.inputs.find((f) => f.kind === 'math' && f.visible)?.el
        || snap.root?.querySelector(SEL.mathField);
      const textField = snap.inputs.find((f) => f.kind === 'text' && f.visible)?.el;

      if (!mathField && !textField) return { ok: false, reason: 'sem campo para expressao' };

      // Alvo MathQuill: dois caminhos, conforme o renderizador do Khan.
      if (mathField && !textField) {
        // O RENDERIZADOR e o criterio do caminho -- nao a presenca do teclado.
        //
        // O Khan mobile MOSTRA um teclado na tela quando o campo e focado, mas
        // ele aparece e some sozinho (re-render, perda de foco). Decidir por
        // "o teclado esta na tela agora?" era uma CORRIDA: a acao comecava pelo
        // teclado, ele sumia no meio, e o campo ficava vazio sem erro nenhum.
        // Era uma falha intermitente (1 em 3) que so aparecia repetindo a mesma
        // acao varias vezes.
        //
        // `.keypad-input` e o campo do celular, e nele os eventos sinteticos
        // funcionam sempre (ver digitarPorEventos). Entao o caminho do teclado
        // na tela fica reservado ao desktop, que e onde ele e estavel.
        const ehCampoMobile = typeof mathField.matches === 'function'
          && (mathField.matches('.keypad-input') || !!mathField.closest('.keypad-input'));

        // (1) Teclado na tela -- e o caminho do DESKTOP.
        if (!ehCampoMobile && openMathKeyboard()) {
          const limpou = await clearMathField(mathField);
          await humanDelay();
          if (!limpou.ok) {
            log(`  aviso: nao consegui limpar o campo (${limpou.motivo})`, 'warn');
          }

          const { ops, unsupported } = latexToMathOps(latex);
          if (!ops.length) {
            return { ok: false, reason: `nada digitavel em "${latex}"${unsupported.length ? ` (faltam: ${unsupported.join('')})` : ''}` };
          }
          const pressed = [];
          await runMathOps(ops, mathField, pressed);
          const got = readMathField(mathField);
          closeMathKeyboard();

          return {
            ok: true,
            detail: `expressao "${latex}" -> ${got || '(vazio)'}`,
            ...(unsupported.length ? { warn: `teclado do Khan nao tem: ${unsupported.join(' ')}` } : {}),
          };
        }

        // (2) SEM teclado na tela -- e o caminho do CELULAR. O Khan mobile usa
        // outro renderizador (.keypad-input) e nao desenha teclado: quem digita
        // e o teclado do sistema, que o agente nao tem. Mas este MathQuill
        // aceita eventos sinteticos -- ver digitarPorEventos, que documenta as
        // duas condicoes (os tres eventos, e a pausa entre caracteres).
        //
        // Aqui o texto vai LINEAR (latexToLinear), e nao como sequencia de
        // teclas do teclado do Khan: nao ha teclado para clicar.
        //
        // Reconsulta o DOM a cada tentativa de proposito. O elemento guardado no
        // snapshot pode ter sido SUBSTITUIDO pelo React entre a observacao e
        // agora -- o proprio apagar dispara re-render. Evento em no destacado
        // nao chega em ninguem, e o campo fica vazio sem erro nenhum: era uma
        // falha intermitente (1 em 3) que so aparecia medindo varias vezes.
        const linear = latexToLinear(latex);
        const campoVivo = () => {
          if (mathField && mathField.isConnected) return mathField;
          const vivos = [...document.querySelectorAll(SEL.mathField)].filter((e) => isVisible(e));
          return vivos.find((e) => e.isConnected) || mathField;
        };
        const lerCampoVivo = () => {
          const vivo = campoVivo();
          return vivo ? readMathField(vivo) : '';
        };

        let got = '';
        for (let tentativa = 1; tentativa <= 3; tentativa++) {
          const alvo = campoVivo();
          await apagarPorEventos(alvo);
          await digitarPorEventos(alvo, linear);
          // Le DEPOIS de assentar. Ler no mesmo instante do ultimo evento pega o
          // estado anterior ao re-render do Khan, e a acao dizia "ok" enquanto o
          // campo aparecia vazio -- uma falha intermitente que so apareceu
          // repetindo a mesma acao varias vezes.
          await sleep(60);
          got = lerCampoVivo();
          if (got) break;
          if (tentativa < 3) {
            log(`  o campo nao registrou (tentativa ${tentativa}/3); tentando de novo.`, 'warn');
            await sleep(200);
          }
        }

        if (!got) {
          return {
            ok: false,
            reason: 'o campo de matematica do celular nao aceitou os eventos sinteticos '
              + '(o campo continua vazio apos 3 tentativas)',
          };
        }
        return {
          ok: true,
          detail: `expressao "${linear}" -> ${got} (eventos sinteticos, sem teclado na tela)`,
          // Nao mentir quando o campo ficou diferente: '^' e o caso conhecido.
          ...(mesmaExpressao(linear, got)
            ? {}
            : { warn: `o campo ficou "${got}", diferente do pedido "${linear}"` }),
        };
      }

      // Campo de texto comum: aqui o setter de value funciona de verdade.
      const linear = latex.includes('\\') ? latexToLinear(latex) : latex;
      setInputValue(textField, linear);
      humanClick(textField);
      return { ok: true, detail: `expressao "${linear}" (campo de texto)` };
    },

    async click_check() {
      const snap = state.snapshot;
      const btn = snap?.buttons?.check;
      if (!btn) return { ok: false, reason: 'sem botao Check' };
      humanClick(btn);
      return { ok: true, detail: 'clicou em Check' };
    },

    async click_next() {
      const snap = state.snapshot;
      const btn = snap?.buttons?.next;
      if (!btn) return { ok: false, reason: 'sem botao Next' };
      humanClick(btn);
      state.epoch++;            // proxima questao: invalida o dedupe
      state.lastFingerprint = null;
      return { ok: true, detail: 'clicou em Next' };
    },

    // O Khan mostra uma tela de "Pronto para praticar? / Vamos lá" antes de
    // comecar a unidade. Sem tratar isso, o agente fica parado olhando um
    // exercicio que ainda nao existe.
    async start_practice() {
      const btn = findStartButton();
      if (!btn) return { ok: false, reason: 'tela de inicio nao encontrada' };
      const label = (btn.innerText || '').trim().slice(0, 30);
      humanClick(btn);
      await humanDelay(800);
      return { ok: true, detail: `clicou em "${label}" para comecar a pratica` };
    },

    async hint() {
      const snap = state.snapshot;
      const btn = snap?.buttons?.hint;
      if (!btn) return { ok: false, reason: 'sem botao de dica' };
      humanClick(btn);
      // O clique sozinho nao servia para nada: o Khan abria a dica e o texto
      // nao voltava para a IA. Agora ele volta no `detail`, entra na memoria da
      // questao (registrarTentativa) e aparece no proximo prompt.
      await humanDelay(500);
      const texto = lerDica();
      return {
        ok: true,
        detail: texto
          ? `dica do Khan: ${texto.slice(0, 220)}`
          : 'clicou em dica (o texto nao foi encontrado na pagina)',
      };
    },

    // Aritmetica exata, sem gastar chamada de IA. Ver calcular() para a
    // gramatica e para o motivo de nao usar eval.
    async calculate(params) {
      const expr = String(params?.expression ?? '').trim();
      if (!expr) return { ok: false, reason: 'params.expression vazio' };
      const r = calcular(expr, params?.vars);
      if (r.erro) return { ok: false, reason: `calculo invalido: ${r.erro}` };
      return { ok: true, detail: `${expr} = ${r.texto}`, valor: r.valor };
    },

    // "Pular" existe no Khan em dois contextos com o MESMO data-testid: pular
    // uma etapa de uma questao em varias partes, e passar de tela. Por isso e
    // uma acao so -- e por isso invalida o dedupe: em qualquer dos dois a
    // pergunta seguinte nao e mais a mesma.
    async skip() {
      const snap = state.snapshot;
      const btn = snap?.buttons?.skip;
      if (!btn) return { ok: false, reason: 'sem botao Pular' };
      humanClick(btn);
      state.epoch++;
      state.lastFingerprint = null;
      return { ok: true, detail: 'clicou em Pular' };
    },

    async ask_user(params) {
      const question = String(params.question || 'Nao consegui responder esta questao. Pode ajudar?');
      state.history.push({ role: 'user', content: `[usuario precisa responder] ${question}` });
      state.aguardandoUsuario = true;
      log(`PERGUNTA PARA O USUARIO: ${question}`, 'ask');
      addChat('assistant', question);
      showPanel(true);
      stopLoop('aguardando usuario. Responda aqui no chat que o agente retoma sozinho.');
      return { ok: true, detail: 'aguardando usuario' };
    },

    async wait(params) {
      const secs = Math.min(Number(params.seconds) || 5, 15);
      await sleep(secs * 1000);
      return { ok: true, detail: `esperou ${secs}s` };
    },
  };

  // Elementos clicaveis das alternativas, na MESMA ordem de extractChoices.

  /* ===========================================================================
   * LOOP PRINCIPAL
   * ======================================================================== */

  // Atraso de re-render do React depois de uma acao.
  //
  // A troca de questao do Khan NAO e atomica: ao clicar em "Proxima pergunta",
  // o contador de progresso e o feedback mudam na hora, mas o enunciado novo so
  // entra no DOM um instante depois. Quem observa nesse meio-termo ve um estado
  // incoerente -- "Calcule 9/m + 4" com prog 2/7 -- e escreve a resposta da
  // pergunta antiga no campo da nova. Nao adianta so aumentar o loopDelayMs: o
  // tempo de re-render varia com o carregamento da pagina.
  //
  // A solucao e esperar a pagina ASSENTAR: so avancar quando duas leituras
  // seguidas concordarem. Se nao assentar no prazo, devolvemos o ultimo
  // snapshot estavel e o loop normal decide.
  //
  // DOIS ajustes de latencia, medidos contra a versao anterior:
  //   1. le ANTES de dormir. A acao anterior ja deixou a pagina assentada, entao
  //      o sleep de proposito atrasava TODA acao mesmo quando ja estava pronta;
  //   2. comeca com 70 ms e afunila. O re-render do React costuma estabilizar
  //      em ~100-200 ms, e os primeiros 220 ms fixos so atrasavam -- o minimo
  //      antes era 440 ms por espera, agora e 70 ms.
  async function waitSettle(attempts = 14, gapMs = 200) {
    let anterior = takeSnapshot();
    for (let i = 0; i < attempts; i++) {
      await sleep(i < 4 ? 70 : gapMs);
      const snap = takeSnapshot();
      if (!snap) { anterior = null; continue; }
      if (anterior && anterior.fingerprint === snap.fingerprint) {
        // Duas leituras iguais: a pagina parou de mudar.
        state.snapshot = snap;
        return snap;
      }
      anterior = snap;
    }
    return anterior;
  }

  function startLoop() {
    if (state.running) return;
    state.running = true;
    // Retomar depois de ask_user consome a espera: a resposta do usuario ja
    // esta no historico e o agente vai agir com ela.
    state.aguardandoUsuario = false;
    state.seen.clear();
    state.epoch++;
    state.lastFingerprint = null;
    state.stuckAdvances = 0;
    state.stuckFingerprint = null;
    state.stallTicks = 0;
    state.stallFingerprint = null;
    log('Agente iniciado.', 'ok');
    paintStatus();
    runLoop();   // fire-and-forget: o proprio loop controla a parada
  }

  function stopLoop(reason = 'Agente parado.') {
    state.running = false;
    log(reason, 'warn');
    paintStatus();
  }

  // Contador de falhas de observacao, para o log explicar o silencio em vez
  // de o agente simplesmente "nao fazer nada" (foi exatamente assim que o
  // defeito dos seletores se manifestou: "Agente iniciado" e mais nada).
  let idleTicks = 0;

  // O exercicio sumiu da tela. Isso so e "pratica acabada" se o contador
  // anterior mostrava 100% -- caso contrario e troca de tela (aula, proxima
  // secao) ou o DOM mudou, e o certo e avisar como problema.
  function isPracticeFinished() {
    const p = state.lastProgress;
    if (!p || !p.total) return false;
    if (p.done < p.total) return false;
    // O contador precisa realmente ter desaparecido: se ainda esta na tela, o
    // que sumiu foi outra coisa e o diagnostico de seletores ainda vale.
    return document.querySelectorAll('[data-testid^="exercise-progress-"]').length === 0;
  }

  async function runLoop() {
    while (state.running) {
      try {
        if (state.busy) { await sleep(state.cfg.loopDelayMs); continue; }

        const snap = takeSnapshot();
        if (!snap) {
          state.snapshot = null;   // sem exercicio na tela: nao age sobre dado velho
          idleTicks++;

          // Fim da lista de exercicios: o contador some e a tela de conclusao
          // aparece. Nao e falha -- e o objetivo. Sem esta checagem, completar
          // a pratica inteira terminava em "Nenhum exercicio encontrado" com
          // um relatorio de seletores, parecendo que tudo deu errado.
          if (idleTicks === 1 && isPracticeFinished()) {
            const p = state.lastProgress;
            log(
              `Pratica concluida: ${p ? p.done : state.solved} de ${p ? p.total : '?'} `
              + 'questoes acertadas. O Khan removeu o contador e mostrou a tela final.',
              'ok'
            );
            addChat('assistant', 'Prática concluída! Todas as questões foram acertadas.');
          }

          // Ainda nao ha exercicio, mas pode ser a tela de inicio da unidade.
          const startBtn = findStartButton();
          if (startBtn && idleTicks === 1) {
            const r = await ACTIONS.start_practice({});
            log(`  ${r.ok ? 'ok' : 'falhou'}: ${r.detail || r.reason}`, r.ok ? 'ok' : 'warn');
            await sleep(1200);
            continue;
          }

          // Fala o que enxerga depois de algumas tentativas, em vez de calar.
          if (idleTicks === 5) {
            log('Nenhum exercicio encontrado na pagina. O que eu enxergo:', 'warn');
            inspectPage().split('\n').forEach((l) => log('  ' + l, 'warn'));
          } else if (idleTicks === 40) {
            log('Continuo sem exercicio. Clique em "Inspecionar" para o relatorio.', 'warn');
            idleTicks = 5;   // repete o diagnostico de tempos em tempos
          }
          await sleep(state.cfg.loopDelayMs);
          continue;
        }
        idleTicks = 0;

        state.snapshot = snap;
        // Guarda o ultimo contador visto: quando o exercicio desaparece, ele e
        // a unica evidencia de como a pratica terminou.
        if (snap.progress) state.lastProgress = snap.progress;

        // Trava de travamento GLOBAL. A tryAdvance so cobre o caminho em que o
        // proprio script desiste da questao; mas a IA pode pedir "click_next"
        // para sempre, o botao aceita o clique e a pagina nao muda -- o agente
        // clica, observa o mesmo enunciado, e repete ate bater no rate limit da
        // API. Aqui contamos observacoes identicas seguidas, independente de quem
        // pediu a acao.
        if (state.stallFingerprint === snap.fingerprint) state.stallTicks += 1;
        else { state.stallFingerprint = snap.fingerprint; state.stallTicks = 0; }

        if (state.stallTicks >= STALL_LIMIT) {
          log(
            `A pagina nao muda ha ${STALL_LIMIT} observacoes seguidas `
            + `(${snap.buttons.next ? 'cliquei em Next e a questao nao mudou' : 'sem botao Next'}). `
            + 'Pode ser o fim da lista de exercicios ou um exercicio travado. '
            + 'Clique em Iniciar para tentar de novo.',
            'err'
          );
          state.stallTicks = 0;
          stopLoop('Agente parado: a pagina parou de responder.');
          return;
        }

        // Dedupe: nao martela a mesma questao. A contagem vive em state.seen.
        const attempts = snap.attempts;
        const alreadyTried = state.lastFingerprint === snap.fingerprint;

        // "Nada mudou" so significa "o React re-renderizou" se NOS nao agimos
        // desde a ultima observacao.
        //
        // Existem acoes que NAO mexem na pagina -- calculate e a principal. Com
        // a regra antiga, depois de calcular a fingerprint era identica e
        // `attempts` continuava 0 (calculate nao e acao de resposta), entao o
        // loop pulava a decisao a cada tique ate a trava global encerrar tudo:
        // o usuario via "usei o calcular e ele nao conseguiu fazer mais nada".
        const agimosNesteFingerprint = state.agiuFingerprint === snap.fingerprint;

        if (alreadyTried && attempts === 0 && !agimosNesteFingerprint) {
          // Mesma questao, nenhuma resposta registrada e nada foi feito: e so
          // re-render do React.
          await sleep(state.cfg.loopDelayMs);
          continue;
        }
        if (attempts >= state.cfg.maxAttemptsPerQuestion) {
          log(`Limite de ${state.cfg.maxAttemptsPerQuestion} tentativas nesta questao. Pulando.`, 'warn');
          await tryAdvance(snap);
          continue;
        }

        state.busy = true;
        state.lastFingerprint = snap.fingerprint;
        paintStatus();

        const plano = await decide(snap);
        if (!plano) { state.busy = false; continue; }

        const result = await performPlano(plano);
        state.busy = false;

        if (!result) { await sleep(state.cfg.loopDelayMs); continue; }
        await waitSettle();
        await sleep(state.cfg.loopDelayMs);
      } catch (err) {
        state.busy = false;
        // Credencial invalida: repetir nao adianta, entao para e orienta.
        if (err.auth) {
          log(`Credencial rejeitada: ${err.message}`, 'err');
          if (err.hint) log(err.hint, 'warn');
          stopLoop('Agente parado por falha de autenticacao.');
          return;
        }
        // Sem saldo/creditos (402): tambem nao adianta repetir -- e a CONTA, nao
        // o pedido. Para e explica o que fazer, em vez de tentar de novo a cada
        // tique ate o usuario desistir sem saber o motivo.
        if (err.payment) {
          log(`A conta do provider esta sem creditos: ${err.message}`, 'err');
          if (err.hint) log(err.hint, 'warn');
          stopLoop('Agente parado: a conta do provider esta sem creditos (HTTP 402).');
          return;
        }
        // Puter sem sessao: o login abre uma janela do navegador, e disparar
        // isso a cada tique do loop seria insistente. O agente para e espera o
        // clique em Config.
        if (err.needPuterLogin) {
          log(err.message, 'warn');
          stopLoop('Agente parado: falta entrar no Puter.');
          return;
        }
        log(`Erro no loop: ${err.message}`, 'err');
        await sleep(state.cfg.loopDelayMs * 2);
      }
    }
  }

  // Pede o plano a IA e normaliza o objeto.
  async function decide(snap) {
    // Questao nova: zera a memoria da anterior. Memoria que ATRAVESSA questoes
    // e ruido -- o modelo passa a "corrigir" a questao atual com o erro de
    // outra, o que e pior do que nao ter memoria nenhuma.
    //
    // A chave e a da QUESTAO (ver chaveDaQuestao), nao a fingerprint: a
    // fingerprint muda quando o aluno erra, e a memoria tem de sobreviver
    // justamente a isso.
    const chave = chaveDaQuestao(snap);
    if (state.plano.chave !== chave) {
      state.plano = { chave, tentativas: [] };
    }

    // Captura o grafico ANTES de montar o prompt: o texto muda conforme a
    // imagem foi anexada ou nao (a IA recebe instrucoes opostas nesses dois
    // casos), entao a ordem importa.
    if (snap.visual && !snap.image && state.cfg.vision && !state.visionDead) {
      state.visionWhy = '';
      snap.image = await captureVisual(snap.root);
      if (snap.image) {
        const kb = Math.round((snap.image.dataUrl.length * 3) / 4 / 1024);
        log(`Grafico capturado: ${snap.image.w}x${snap.image.h} ${snap.image.kind}, ${kb} KB`, 'ok');
      } else if (state.cfg.vision) {
        log(`Questao com grafico, mas nao consegui extrair a imagem (${state.visionWhy}).`, 'warn');
      }
    }

    // Alternativas que sao SO imagem ("qual destes graficos..."). Sem isto a IA
    // recebe "[0] ", "[1] " e escolhe no escuro -- ver capturarAlternativasSemTexto.
    snap.imagensAlternativas = [];
    if (snap.choices.some((c) => !(c.text || '').trim() && c.temImagem)) {
      snap.imagensAlternativas = await capturarAlternativasSemTexto(snap.root, snap.choices);
      if (snap.imagensAlternativas.length) {
        log(
          `Alternativas com imagem: ${snap.imagensAlternativas.length} capturada(s) `
          + `(indices ${snap.imagensAlternativas.map((i) => i.index).join(', ')}).`,
          'ok'
        );
      } else if (!state.visionDead && !state.extrasDead) {
        log('Ha alternativa sem texto, mas nao consegui capturar a figura dela.', 'warn');
      }
    }

    let prompt = buildUserPrompt(snap);
    // Antes de gastar uma chamada de IA numa questao sem controle nenhum, dizer
    // o que a pagina tem -- ver diagnosticarSemControles.
    if (!snap.choices.length && !snap.inputs.length) diagnosticarSemControles(snap);
    log(
      `Observou: "${snap.problem.slice(0, 60).replace(/\n/g, ' ')}" ` +
      `| ${snap.choices.length} alt | ${snap.inputs.length} campos` +
      (snap.inputs.filter((f) => f.kind === 'math').length
        ? ` (${snap.inputs.filter((f) => f.kind === 'math').length} math)`
        : '') +
      ' | ' +
      `check:${snap.hasCheck ? 's' : 'n'} next:${snap.hasNext ? 's' : 'n'}` +
      (snap.progress ? ` | prog ${snap.progress.done}/${snap.progress.total} ${snap.progress.state}` : '') +
      (snap.feedback ? ` | feedback: ${snap.feedback}` : '')
    );

    // Chamada com negociacao de formato.
    //
    // Cada provider tem suas manias e nenhuma documenta todas:
    //   - so 1 mensagem por request (NVIDIA NIM /vlm/...): o prompt de
    //     sistema tem de entrar dentro da mensagem de usuario;
    //   - image_url como objeto (OpenAI, Groq) ou como string (NVIDIA);
    //   - modelo sem visao: a parte de imagem simplesmente e recusada.
    //
    // Sao ajustes INDEPENDENTES, e a ordem em que a API os acusa depende da
    // validacao dela -- por isso e laco, nao if/else. Cada iteracao conserta
    // um ponto e reenvia; o limite evita laco se algum erro nao for
    // reconhecido.
    let raw = null;
    for (let tentativa = 1; tentativa <= 4; tentativa++) {
      try {
        const t0IA = Date.now();
        raw = await askAI(SYSTEM_PROMPT, prompt, snap.image, snap.imagensAlternativas);
        // Quanto tempo a IA levou, para a linha de tempo do plano mostrar a
        // fatia real. Num provider rapido (Groq ~1s) ela e quase tudo; no
        // Gemini pode ser 8s e as esperas locais somem.
        state.msIA = Date.now() - t0IA;
        break;
      } catch (err) {
        if (err.singleMessageWrong && !state.singleMessage) {
          state.singleMessage = true;
          log('A API aceita so 1 mensagem. Reunindo sistema e estado em uma.', 'warn');
          continue;
        }
        if (err.imgStyleWrong && state.imgStyle !== 'string') {
          state.imgStyle = 'string';
          log('A API quer image_url como texto, nao como objeto. Reenviando.', 'warn');
          continue;
        }
        if (err.visionUnsupported) {
          // Modelo sem visao: repetir o mesmo corpo so queima cota. Segue com
          // o texto e marca o modelo para nao tentar imagem de novo.
          log(`${err.message} Seguindo sem imagem.`, 'warn');
          state.visionDead = true;
          snap.image = null;
          prompt = buildUserPrompt(snap);
          continue;
        }
        // Tool calling recusado: desce um degrau e reenvia. O JSON no texto
        // continua sendo o caminho de quem nao aceita tools, entao nada quebra
        // -- so perde a vantagem do schema.
        if (err.toolChoiceWrong && !state.toolChoiceDead) {
          state.toolChoiceDead = true;
          log('A API aceita tools mas recusou tool_choice. Reenviando sem ele.', 'warn');
          continue;
        }
        if (err.toolsUnsupported && !state.toolsDead) {
          state.toolsDead = true;
          log('A API nao aceita tool calling. Seguindo com JSON no texto.', 'warn');
          continue;
        }
        // Endpoint que so aceita UMA imagem: perde as alternativas-imagem, mas
        // mantem a visao da figura principal. Descartar as duas coisas seria
        // pior -- a principal e o enunciado da questao.
        if (err.tooManyImages && !state.extrasDead) {
          state.extrasDead = true;
          snap.imagensAlternativas = [];
          log('A API aceita so uma imagem por chamada. Seguindo sem as alternativas-imagem.', 'warn');
          continue;
        }
        throw err;
      }
    }
    if (raw == null) throw new Error('A IA recusou o formato da requisicao.');
    const plan = parseJsonLoose(raw);

    // Normaliza UM action e {"actions":[...]} para a MESMA lista de passos.
    //
    // A partir daqui so existe o formato de lista: o loop, a memoria e o
    // performPlano nao precisam saber qual dos dois a IA mandou. Sem isso,
    // haveria dois caminhos de execucao para manter em sincronia -- e o modo
    // tool calling (que chega como "actions" quando vem mais de uma chamada)
    // divergiria do modo texto.
    const passos = Array.isArray(plan?.actions)
      ? plan.actions
      : (plan && typeof plan.action === 'string' ? [plan] : null);
    if (!passos || !passos.length) {
      log('Plano invalido recebido da IA (sem "action" nem "actions").', 'err');
      return null;
    }
    if (passos.length > state.cfg.maxPlanSteps) {
      log(`Plano com ${passos.length} passos; usando os ${state.cfg.maxPlanSteps} primeiros.`, 'warn');
      passos.length = state.cfg.maxPlanSteps;
    }

    // O que o modelo DISSE nesta resposta, seja por tool calling (content /
    // reasoning_content) ou pelo campo `thought` do JSON.
    //
    // O descarte do `raw` importa: quando a resposta vem como JSON puro
    // (caminho de texto), a "narracao" E o proprio JSON do plano, e usa-la como
    // frase do modelo sujava o log com {"action":"wait","params":{}}. Nesse
    // caso quem carrega o texto do modelo e o campo `thought` do JSON.
    const narracao = (state.narracao && state.narracao.trim() !== String(raw).trim())
      ? state.narracao
      : '';

    const acoes = [];
    for (const p of passos) {
      const action = String(p?.action ?? '').trim();
      if (!ACTIONS[action]) {
        log(`Acao desconhecida da IA: "${action}".`, 'err');
        // Passo invalido no INICIO invalida o plano inteiro. No MEIO, executamos
        // o que ja veio e paramos ali -- descartar o começo bom por causa do fim
        // ruim desperdicaria a chamada.
        if (!acoes.length) return null;
        break;
      }
      // `thought` viaja DENTRO dos params quando a resposta veio por tool
      // calling (ver toolSchemas). Ele e a EXPLICACAO, nao um parametro da
      // acao -- por isso sai dos params antes de executar, para nao virar lixo
      // no lugar de um campo de verdade.
      const params = { ...(p.params || p.action_params || {}) };
      const dito = params.thought;
      delete params.thought;
      if (action === 'select_choice') conferirChoiceText(params, snap);
      acoes.push({
        action,
        params,
        thought: String(dito || p.thought || plan.thought || narracao || '').trim(),
      });
    }
    if (!acoes.length) return null;
    return {
      acoes,
      thought: acoes[0].thought,
      // A narracao completa fica no plano para o log mostrar UMA vez, em vez de
      // repetir a cada passo.
      narracao,
    };
  }

  // Conference o indice contra o texto que a IA disse ter escolhido.
  //
  // O erro que mais aparece em multipla escolha NAO e a matematica: e o
  // indice. O modelo raciocina certo ("crescente ate x=2, ou seja ]-inf,2["),
  // escreve o intervalo certo no raciocinio e mesmo assim manda
  // {"choice_index":1}, que aponta para outra alternativa. Ele ate descreve a
  // resposta que pretendia, entao a correcao e mecanica e nao adivinhacao.
  //
  // Por isso o pedido de choice_text: o modelo escreve a alternativa escolhida
  // em palavras, e aqui conferimos que o indice bate com as palavras. Se nao
  // bater, procuramos o texto entre as alternativas e usamos o indice que
  // corresponde. Nao e a IA decidindo de novo -- e a maquina impedindo que o
  // clique contradiga a frase que o proprio modelo escreveu.
  function normEscolha(s) {
    // O Khan renderiza "] 1 , 3 ["; a IA escreve "]1, 3[", ou " ] -inf, 2 [",
    // ou "1 a 3". Espaco e pontuacao nao mudam a resposta, entao somem.
    // Simbolos, ordem e texto continuam valendo.
    return String(s == null ? '' : s)
      .toLowerCase()
      .normalize('NFKC')
      .replace(/[\u2212\u2013\u2014]/g, '-')  // sinonimos de menos
      .replace(/\u221e/g, 'inf')              // o modelo escreve "inf", o DOM "∞"
      .replace(/[\s[\](){}|<>]/g, '')          // espacos e chaves decorativas
      .replace(/[.,;:]+$/g, '');               // pontuacao final
  }

  function conferirChoiceText(params, snap) {
    const alvos = snap?.choices || [];
    const idx = Number(params.choice_index);
    const dito = params.choice_text;

    if (!alvos.length || !dito || !Number.isInteger(idx)) return;

    const alvo = alvos[idx];
    if (alvo && normEscolha(alvo.text) === normEscolha(dito)) return; // bateu

    const casam = alvos
      .map((c, i) => ({ i, ok: normEscolha(c.text) === normEscolha(dito) }))
      .filter((x) => x.ok);

    // UMA alternativa casa: e a que a IA quis dizer, mesmo que com pontuacao
    // diferente. Corrigimos o indice.
    //
    // Mais de uma casa NAO corrige: em matematica "]1,2[" e "[1,2[" (aberto e
    // fechado) diferem so nos colchetes, que a normalizacao apaga. Escolher
    // entre elas seria chute nosso -- pior que o erro da IA.
    if (casam.length === 1) {
      const { i } = casam[0];
      log(
        `Indice ${idx} nao bate com o texto dito ("${dito}" = [${i}] "${alvos[i].text}"). `
        + 'Corrigindo o indice.',
        'warn'
      );
      params.choice_index = i;
      return;
    }

    if (casam.length > 1) {
      log(
        `choice_text "${dito}" e ambiguo: casa com [${casam.map((c) => c.i).join('], [')}] `
        + `(diferem so por pontuacao). Mantendo o indice ${idx}.`,
        'warn'
      );
      return;
    }

    // Nenhuma casa: parafrase. Nao corrige nada, mas deixa registrado -- o log
    // e a unica forma de ver isso depois.
    log(
      `choice_text "${dito}" nao bate com nenhuma alternativa (indice ${idx}: `
      + `"${alvo ? alvo.text : 'inexistente'}"). Pode ser parafrase.`,
      'warn'
    );
  }

  // Executa o plano e atualiza contadores.
  const ANSWER_ACTIONS = new Set(['select_choice', 'type_answer', 'build_expression']);

  // O campo de TEXTO vivo no indice dado.
  //
  // Reextrai o formulario em vez de confiar no elemento guardado no snapshot: o
  // React troca o no entre a observacao e a acao, e escrever no no antigo nao
  // muda nada na tela.
  function fieldDeTextoVivo(indice, fallback) {
    const root = state.snapshot?.root;
    if (root) {
      const alvo = extractInputs(root)[indice];
      if (alvo && alvo.kind !== 'math' && alvo.el?.isConnected) return alvo.el;
    }
    return fallback?.el?.isConnected ? fallback.el : null;
  }

  // O valor que o campo de texto no indice dado TEM agora, lido da pagina viva.
  function valorNoCampoVivo(indice) {
    const root = state.snapshot?.root;
    if (!root) return null;
    const alvo = extractInputs(root)[indice];
    return alvo?.el ? String(alvo.el.value ?? '').trim() : null;
  }

  // Campos de resposta VISIVEIS que ainda estao vazios.
  //
  // Existe por causa das questoes de VARIOS campos (preencher uma tabela, por
  // exemplo). Nelas, conferir depois de preencher so o primeiro campo marca a
  // resposta PARCIAL como errada -- e a tentativa e contada, entao a questao se
  // perde. Foi o sintoma relatado: "nao consegue responder mais de um campo".
  function camposVazios(snap) {
    return (snap?.inputs || []).filter((f) => {
      if (!f.visible) return false;
      const valor = f.kind === 'math'
        ? f.value
        : (f.el && 'value' in f.el ? f.el.value : f.value);
      return !String(valor ?? '').trim();
    });
  }

  async function perform(plan) {
    const fn = ACTIONS[plan.action];
    if (!fn) return { ok: false, reason: `acao desconhecida "${plan.action}"` };
    // A frase do modelo vai na propria linha da acao: e o que permite conferir
    // o que ele PRETENDIA antes de ver o efeito.
    log(`> ${plan.action} :: ${String(plan.thought || '').slice(0, 220)}`.trim(), 'action');

    const snapAtPlan = state.snapshot;
    const result = await fn(plan.params);

    if (result?.ok) {
      log(`  ok: ${result.detail}`, 'ok');
      state.solved++;
      if (ANSWER_ACTIONS.has(plan.action)) {
        // Conta a tentativa para o dedupe nao reencostar na mesma questao.
        //
        // Mas so se a resposta estiver COMPLETA. Numa questao de varios campos,
        // preencher um por vez nao e uma tentativa -- a resposta ainda esta
        // incompleta. Contar cada campo esgotava o limite (2 tentativas) antes
        // de a questao ser respondida, e o agente pulava a questao sozinho.
        const fp = snapAtPlan?.fingerprint;
        const depois = takeSnapshot();
        const vazios = depois ? camposVazios(depois) : [];
        if (fp && !vazios.length) {
          state.seen.set(fp, (state.seen.get(fp) || 0) + 1);
        } else if (vazios.length) {
          log(`  resposta incompleta: ${vazios.length} campo(s) ainda vazio(s); nao conta como tentativa.`, 'info');
        }
      }
    } else {
      state.wrong++;
      log(`  falhou: ${result?.reason || 'motivo desconhecido'}`, 'err');
    }
    paintStatus();
    // Marca que HOUVE acao neste estado da pagina. E o que distingue "o React
    // re-renderizou" de "eu agi e a pagina nao mudou" -- ver o dedupe no loop.
    if (snapAtPlan?.fingerprint) state.agiuFingerprint = snapAtPlan.fingerprint;
    registrarTentativa(plan, result, snapAtPlan);
    return result;
  }

  // Grava o que o agente tentou NESTA questao, para o proximo prompt poder
  // dizer "voce ja tentou isso e falhou".
  //
  // E a diferenca entre CORRIGIR e resolver de novo do zero -- e, na pratica,
  // o que impede o modelo de marcar a MESMA alternativa que acabou de errar
  // (antes nao havia registro nenhum dela).
  function registrarTentativa(plan, result, snapAtPlan) {
    if (!state.plano) state.plano = { chave: snapAtPlan ? chaveDaQuestao(snapAtPlan) : null, tentativas: [] };
    // A tentativa pertence a questao do snapshot em que foi decidida. Se a
    // pagina ja trocou (o click_next da propria acao faz isso), ela nao
    // interessa mais e sujaria a memoria da proxima questao.
    if (snapAtPlan && state.plano.chave && state.plano.chave !== chaveDaQuestao(snapAtPlan)) return;
    state.plano.tentativas.push({
      action: plan.action,
      params: plan.params,
      thought: plan.thought,
      ok: !!result?.ok,
      detail: result?.ok ? result.detail : result?.reason,
      feedback: snapAtPlan?.feedback || null,
    });
    // Teto: numa questao que ja errou muitas vezes o prompt nao pode crescer
    // sem limite. As ULTIMAS tentativas sao as que importam.
    if (state.plano.tentativas.length > 8) state.plano.tentativas.shift();
  }

  // O `ok` de uma acao de resposta diz que a MECANICA funcionou (o campo foi
  // preenchido, a alternativa foi marcada), NAO que a resposta esta certa. O
  // veredito so aparece depois do Check.
  //
  // Sem isto a memoria dizia "OK" para uma resposta errada -- e o modelo lia
  // isso como "pode seguir", que e o oposto do que a memoria existe para fazer.
  async function marcarResultadoDaResposta() {
    // Espera o estado MUDAR, em vez de um tempo fixo. Um sleep fixo erra para
    // os dois lados: curto demais le o estado anterior (e a memoria grava "OK"
    // numa resposta errada), longo demais atrasa o loop em toda questao.
    //
    // Comeca rapido (60 ms) e afunila: o Khan atualiza o progresso em ~100-300
    // ms na maioria das vezes, e os 150 ms fixos de antes atrasavam toda
    // questao mesmo quando o estado ja estava la.
    const inicial = readProgress()?.state;
    let prog = null;
    for (let i = 0; i < 16; i++) {
      await sleep(i < 4 ? 60 : 150);
      prog = readProgress();
      if (prog?.state === 'correct' || prog?.state === 'incorrect') break;
      if (prog?.state && prog.state !== inicial) break;
    }
    const t = [...state.plano.tentativas].reverse().find((x) => ANSWER_ACTIONS.has(x.action));
    if (!t) return;
    prog = prog || readProgress();
    if (prog?.state === 'correct') t.acertou = true;
    if (prog?.state === 'incorrect') t.errou = true;
    const fb = extractFeedback(state.snapshot?.root);
    if (fb) t.feedback = fb;
  }

  // Executa o plano -- 1 passo ou N -- com re-observacao BARATA entre eles.
  //
  // O ganho: responder + verificar + avancar numa UNICA chamada de IA. A
  // re-observacao entre passos e local (takeSnapshot), nao gasta cota, e e o
  // que permite abortar quando um passo deixa de se aplicar: se a resposta
  // errou, nao existe "Continuar", e o plano para ali sozinho.
  async function performPlano(plano) {
    const passos = plano.acoes || [];
    // Cronometro do plano. Ver logTempoPlano para o que esta medicao responde.
    const t0 = Date.now();
    let msExec = 0;

    // O que o modelo DISSE antes de agir. Quando a resposta veio por tool
    // calling nao ha campo de texto proprio, entao esta e a unica janela para a
    // intencao -- foi por isso que o log ficava "> select_choice ::" vazio.
    //
    // So aparece separado quando ACRESCENTA algo: se a frase ja e o `thought`
    // do primeiro passo, ela sai na linha da propria acao.
    const narracao = String(plano.narracao || '').trim();
    if (narracao && narracao !== String(passos[0]?.thought || '').trim()) {
      log(`modelo: ${narracao.slice(0, 600)}`, 'info');
    }
    // autoCheck so age quando o plano NAO pediu o check. Sem esta regra,
    // {"actions":[select_choice, click_check]} clicaria em Verificar duas vezes
    // -- e o proprio prompt proibe isso.
    const planoJaCheca = passos.some((p) => p.action === 'click_check');
    let ultimo = null;

    for (let i = 0; i < passos.length; i++) {
      if (!state.running) break;
      const passo = passos[i];

      if (i > 0) {
        // Entre passos a pagina precisa assentar (o React re-renderiza) e a
        // pre-condicao precisa ser reconferida contra o estado NOVO -- e nao
        // contra o snapshot em que o plano foi decidido.
        await waitSettle();
        const s = takeSnapshot();
        if (s) state.snapshot = s;
      }

      const disponivel = precondicaoDe(passo.action);
      if (!disponivel.ok) {
        log(
          `  passo ${i + 1} (${passo.action}) nao se aplica: ${disponivel.reason}. `
          + 'Plano encerrado aqui.',
          'warn'
        );
        break;
      }

      const tAcao = Date.now();
      ultimo = await perform(passo);
      msExec += Date.now() - tAcao;
      if (!ultimo?.ok) break;

      // O Check e quem produz o veredito da resposta: e aqui que a memoria
      // deixa de dizer "preenchi o campo" e passa a dizer "acertei/errei".
      if (passo.action === 'click_check') {
        const tCheck = Date.now();
        await marcarResultadoDaResposta();
        msExec += Date.now() - tCheck;
      }

      // Resposta enviada e o plano nao mandou verificar: o passo natural
      // seguinte e conferir, entao o script faz isso por conta propria.
      if (!planoJaCheca && ANSWER_ACTIONS.has(passo.action) && state.cfg.autoCheck) {
        const s = takeSnapshot();
        if (s) state.snapshot = s;
        const vazios = camposVazios(state.snapshot);
        if (state.snapshot?.buttons?.check && vazios.length) {
          // Nao verificar com campo obrigatorio vazio: o Khan marcaria a
          // resposta PARCIAL como errada. Numa questao de varios campos, a IA
          // pode preencher um por vez -- e o prompt da lista de campos com
          // "valor_atual", entao ela sabe o que falta.
          log(`  ainda ha ${vazios.length} campo(s) vazio(s): nao vou verificar agora.`, 'warn');
        } else if (state.snapshot?.buttons?.check) {
          const tAuto = Date.now();
          await humanDelay();
          const c = await ACTIONS.click_check({});
          msExec += Date.now() - tAuto;
          log(`  ${c.ok ? 'ok' : 'falhou'}: ${c.detail || c.reason}`, c.ok ? 'ok' : 'warn');
          if (!c.ok) break;
          const tAuto2 = Date.now();
          await marcarResultadoDaResposta();
          msExec += Date.now() - tAuto2;
        }
      }
    }
    logTempoPlano(t0, msExec);
    return ultimo;
  }

  // Uma linha por plano, com o tempo que ele consumiu.
  //
  // A duvida que deu origem a isto era "por que demora tanto quando a resposta
  // vem da IA?", e sem numero a resposta e palpite. A linha separa as tres
  // fatias -- latencia da IA, execucao das acoes e o resto (esperas locais de
  // assentamento e o intervalo do loop):
  //
  //   plano em 8.4s  (IA 7.1s, acoes 0.4s, esperas 0.9s)
  //
  // Se a IA domina, o provider e o gargalo e mexer nas esperas nao adianta. Se as
  // esperas dominam, ai sim vale otimizar o sleep. A medicao escolhe o caminho.
  function logTempoPlano(t0, execMs) {
    const total = Date.now() - t0;
    const espera = Math.max(0, total - state.msIA - execMs);
    log(
      `plano em ${(total / 1000).toFixed(1)}s`
      + `  (IA ${(state.msIA / 1000).toFixed(1)}s, `
      + `acoes ${(execMs / 1000).toFixed(1)}s, `
      + `esperas ${(espera / 1000).toFixed(1)}s)`,
      'info'
    );
  }

  // A pre-condicao de uma acao contra o snapshot ATUAL. Vem do catalogo, entao
  // a regra de "quando isto faz sentido" existe num lugar so -- a mesma que
  // monta a lista "ACOES DISPONIVEIS" que a IA recebe.
  function precondicaoDe(nome) {
    const t = achaTool(nome);
    if (!t) return { ok: false, reason: 'acao desconhecida' };
    const s = state.snapshot;
    if (!s) return { ok: false, reason: 'sem snapshot da pagina' };
    try {
      return t.available(s)
        ? { ok: true }
        : { ok: false, reason: 'a pagina nao tem mais o que ela precisa' };
    } catch (e) {
      return { ok: false, reason: `erro ao avaliar: ${e.message}` };
    }
  }

  // Avanca quando a questao travou (limite de tentativas esgotado).
  //
  // Botao Next que nao muda de problema (fim da lista, exercicio travado,
  // clique que o React ignorou) fazia o agente repetir "Avancando questao
  // travada" para sempre. Aqui contamos as avancadas sem troca real de
  // fingerprint e, se passar do limite, paramos com explicacao em vez de
  // martelar a pagina.
  async function tryAdvance(snap) {
    if (!state.running) return;
    const s = snap || state.snapshot || takeSnapshot();

    if (state.stuckAdvances === undefined) state.stuckAdvances = 0;
    if (state.stuckFingerprint === s?.fingerprint) state.stuckAdvances += 1;
    else { state.stuckFingerprint = s?.fingerprint; state.stuckAdvances = 1; }

    if (state.stuckAdvances > 3) {
      log(
        `Nao consegui avancar apos ${state.stuckAdvances} tentativas. `
        + 'O botao Next nao muda de questao -- pode ser o fim da lista ou '
        + 'um exercicio travado. Clique em Iniciar para tentar de novo.',
        'err'
      );
      state.stuckAdvances = 0;
      stopLoop('Agente parado: nao foi possivel avancar.');
      return;
    }

    if (s?.buttons?.next) {
      log(`Avancando questao travada (${state.stuckAdvances}/3).`, 'warn');
      await ACTIONS.click_next({});
    } else {
      // Sem Next visivel: recarrega a secao de exercicio sem perder a sessao.
      log('Sem botao Next: recarregando o exercicio.', 'warn');
      const exercise = findExerciseRoot();
      if (exercise) {
        exercise.scrollIntoView({ block: 'center', behavior: 'smooth' });
        await sleep(600);
      }
    }
    await sleep(state.cfg.loopDelayMs * 2);
  }

  /* ===========================================================================
   * PAINEL FLUTUANTE
   * ======================================================================== */
  const HOST_ID = 'khan-agent-root';
  let ui = {};

  /* ===========================================================================
   * DISPOSITIVO: PC x CELULAR
   * ======================================================================== */
  // O painel nasceu para desktop: 380x460 fixos no canto, fonte 11px, botoes de
  // 4px de padding. Num celular isso da tres problemas: cobre metade da questao,
  // os alvos de toque ficam menores que o dedo (o minimo confortavel e ~44px) e
  // a fonte de 12px em input faz o Android dar zoom ao focar.
  //
  // A deteccao combina tres sinais, de proposito:
  //   - UA de celular/tablet  -> pega o caso obvio, inclusive tablet grande;
  //   - largura da janela     -> pega janela estreita no PC (util de verdade);
  //   - ponteiro grosso       -> so AUMENTA os alvos de toque, sem trocar o
  //                              layout. Usar isso para decidir layout daria
  //                              falso positivo em notebook com tela sensivel.
  const LARGURA_MOBILE = 820;

  function ehToque() {
    try { return window.matchMedia('(pointer: coarse)').matches; } catch { return false; }
  }

  function detectarDispositivo() {
    const ua = navigator.userAgent || '';
    const porUA = /android|iphone|ipad|ipod|iemobile|opera mini|mobile/i.test(ua)
      // userAgentData e o caminho novo e nao depende de string fragil.
      || navigator.userAgentData?.mobile === true;
    const porTamanho = window.innerWidth <= LARGURA_MOBILE;
    return (porUA || porTamanho) ? 'mobile' : 'desktop';
  }

  // Aplica a classe no host e reavalia quando a tela muda. Sem o listener, girar
  // o celular ou redimensionar a janela deixaria o layout no modo antigo ate
  // recarregar a pagina.
  function aplicarDispositivo() {
    if (!ui.host) return null;
    const d = detectarDispositivo();
    const toque = ehToque();
    ui.host.classList.toggle('ka-mobile', d === 'mobile');
    ui.host.classList.toggle('ka-desktop', d === 'desktop');
    ui.host.classList.toggle('ka-touch', toque || d === 'mobile');
    ui.device = d;
    return d;
  }

  function vigiarDispositivo() {
    let t = null;
    const reavaliar = () => {
      clearTimeout(t);
      // Debounce: o resize dispara dezenas de vezes durante o gesto, e mexer no
      // DOM a cada evento engasga a animacao.
      t = setTimeout(() => {
        const antes = ui.device;
        const agora = aplicarDispositivo();
        if (antes && agora && antes !== agora) {
          log(`Layout ajustado para ${agora === 'mobile' ? 'celular' : 'computador'}.`, 'info');
        }
      }, 180);
    };
    window.addEventListener('resize', reavaliar, { passive: true });
    window.addEventListener('orientationchange', reavaliar, { passive: true });
  }

  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      #${HOST_ID} { all: initial; }
      #${HOST_ID}, #${HOST_ID} * { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
      #${HOST_ID} .ka-panel {
        position: fixed; right: 16px; bottom: 16px; width: 380px; max-width: calc(100vw - 32px);
        height: 460px; max-height: calc(100vh - 32px);
        background: #12151c; color: #e6e9ef; border: 1px solid #2a3040; border-radius: 12px;
        box-shadow: 0 18px 48px rgba(0,0,0,.45); display: flex; flex-direction: column;
        overflow: hidden; z-index: 2147483000; font-size: 13px; line-height: 1.45;
      }
      #${HOST_ID} .ka-head {
        display: flex; align-items: center; gap: 8px; padding: 10px 12px;
        background: #1a1f2b; border-bottom: 1px solid #2a3040; cursor: grab; user-select: none;
        /* Sem isto o gesto de arrastar no celular vira scroll da pagina: o
           navegador trata o movimento como pane antes de entregar o
           pointermove. O cabecalho e a alca, entao ele nao rola. */
        touch-action: none;
      }
      #${HOST_ID} .ka-head:active { cursor: grabbing; }
      /* Os grupos do cabecalho so agrupam no celular. Aqui eles somem da
         arvore de layout (display:contents) e os filhos entram no flex do
         cabecalho na ordem do HTML -- que e exatamente a ordem original do
         desktop. Sem isso o desktop mudaria junto. */
      #${HOST_ID} .ka-head > .ka-h1,
      #${HOST_ID} .ka-head > .ka-h2,
      #${HOST_ID} .ka-head > .ka-h3 { display: contents; }
      #${HOST_ID} .ka-dot { width: 9px; height: 9px; border-radius: 50%; background: #6b7280; flex: none; }
      #${HOST_ID} .ka-dot.on { background: #22c55e; box-shadow: 0 0 8px #22c55e88; }
      #${HOST_ID} .ka-dot.busy { background: #f59e0b; box-shadow: 0 0 8px #f59e0b88; }
      #${HOST_ID} .ka-title { font-weight: 600; font-size: 13px; flex: 1; }
      #${HOST_ID} .ka-stat { color: #8b93a7; font-size: 11px; font-variant-numeric: tabular-nums; }
      #${HOST_ID} .ka-btn {
        background: #232a38; color: #cfd6e4; border: 1px solid #333c4e; border-radius: 6px;
        padding: 4px 9px; font-size: 11px; cursor: pointer; line-height: 1.4;
      }
      #${HOST_ID} .ka-btn:hover { background: #2c3446; }
      #${HOST_ID} .ka-btn.go { background: #166534; border-color: #22c55e; color: #dcfce7; }
      #${HOST_ID} .ka-btn.stop { background: #7f1d1d; border-color: #ef4444; color: #fee2e2; }
      #${HOST_ID} .ka-body { flex: 1; display: flex; flex-direction: column; min-height: 0; }
      #${HOST_ID} .ka-log { flex: 1; overflow-y: auto; padding: 8px 10px; font-size: 11.5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      #${HOST_ID} .ka-log > div { margin-bottom: 3px; word-break: break-word; }
      #${HOST_ID} .ka-t { color: #5c6478; }
      #${HOST_ID} .ka-info { color: #a7b0c0; }
      #${HOST_ID} .ka-ok { color: #4ade80; }
      #${HOST_ID} .ka-warn { color: #fbbf24; }
      #${HOST_ID} .ka-err { color: #f87171; }
      #${HOST_ID} .ka-action { color: #7dd3fc; }
      #${HOST_ID} .ka-ask { color: #c084fc; }
      #${HOST_ID} .ka-input { display: flex; gap: 6px; padding: 8px; border-top: 1px solid #2a3040; }
      #${HOST_ID} .ka-input textarea {
        flex: 1; resize: none; background: #0d1017; color: #e6e9ef; border: 1px solid #333c4e;
        border-radius: 6px; padding: 6px 8px; font-size: 12px; min-height: 34px; max-height: 90px;
        font-family: inherit; outline: none;
      }
      #${HOST_ID} .ka-input textarea:focus { border-color: #3b82f6; }
      #${HOST_ID} .ka-min .ka-body { display: none; }
      #${HOST_ID} .ka-min { height: auto; }

      /* ---- tela de configuracao ---- */
      #${HOST_ID} .ka-cfg { display: flex; flex-direction: column; flex: 1; min-height: 0; border-top: 1px solid #2a3040; background: #0f1219; }
      #${HOST_ID} .ka-cfg[hidden] { display: none; }
      #${HOST_ID} .ka-cfg-scroll { flex: 1; overflow-y: auto; padding: 12px; }
      #${HOST_ID} .ka-lbl { display: block; font-size: 11px; color: #8b93a7; margin: 10px 0 4px; font-weight: 600; letter-spacing: .02em; }
      #${HOST_ID} .ka-lbl:first-child { margin-top: 0; }
      #${HOST_ID} .ka-in {
        width: 100%; background: #0a0d13; color: #e6e9ef; border: 1px solid #333c4e;
        border-radius: 6px; padding: 6px 8px; font-size: 12px; font-family: inherit; outline: none;
      }
      #${HOST_ID} .ka-in:focus { border-color: #3b82f6; }
      #${HOST_ID} .ka-row { display: flex; gap: 6px; align-items: center; }
      #${HOST_ID} .ka-row .ka-in { flex: 1; }
      #${HOST_ID} .ka-hint { margin: 5px 0 0; font-size: 10.5px; color: #6b7488; line-height: 1.4; }
      #${HOST_ID} .ka-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 10px; }
      #${HOST_ID} .ka-check { display: flex; align-items: center; gap: 7px; margin-top: 12px; font-size: 12px; color: #cfd6e4; cursor: pointer; }
      #${HOST_ID} .ka-puterbox { margin: 6px 0 10px; padding: 8px 10px; border: 1px solid #2b3242; border-radius: 7px; background: #171c26; }
      #${HOST_ID} .ka-puterbox .ka-btn { margin-top: 7px; }
      #${HOST_ID} .ka-cfg-foot {
        display: flex; gap: 6px; align-items: center; padding: 9px 12px;
        border-top: 1px solid #2a3040; background: #141926;
      }
      #${HOST_ID} .ka-cfg-msg { flex: 1; font-size: 11px; line-height: 1.35; }
      #${HOST_ID} .ka-cfg-msg.ok { color: #4ade80; }
      #${HOST_ID} .ka-cfg-msg.err { color: #f87171; }
      #${HOST_ID} .ka-cfg-msg.busy { color: #fbbf24; }
      /* Catalogo de modelos: lista rolavel com filtro. Os chips de capacidade
         so aparecem quando o provider DECLARA a capacidade -- ver
         normalizarModelos. */
      #${HOST_ID} .ka-models {
        max-height: 150px; overflow-y: auto; margin: 6px 0 2px;
        border: 1px solid #2a3040; border-radius: 8px; background: #0e1116;
      }
      #${HOST_ID} .ka-models-head {
        position: sticky; top: 0; z-index: 1; background: #151a23; color: #8b93a7;
        font-size: 10.5px; padding: 4px 8px; border-bottom: 1px solid #2a3040;
      }
      #${HOST_ID} .ka-model {
        display: flex; align-items: center; gap: 6px; width: 100%; text-align: left;
        background: none; border: none; border-bottom: 1px solid #1c222d;
        color: #cfd6e4; font-size: 11.5px; padding: 6px 8px; cursor: pointer;
      }
      #${HOST_ID} .ka-model:last-child { border-bottom: none; }
      #${HOST_ID} .ka-model:hover { background: #1b2130; }
      #${HOST_ID} .ka-model-id {
        flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      }
      #${HOST_ID} .ka-chip {
        flex: none; font-size: 9.5px; padding: 1px 5px; border-radius: 999px;
        border: 1px solid #3b4658; color: #9aa4b8;
      }
      #${HOST_ID} .ka-chip.visao { border-color: #2f6b45; color: #7ee0a5; }
      #${HOST_ID} .ka-chip.tools { border-color: #3f5b8a; color: #93b8f0; }
      #${HOST_ID} .ka-keydot { color: #fbbf24; font-size: 10px; }
      #${HOST_ID} .ka-keydot.on { color: #4ade80; }

      /* ===================================================================
       * MODO CELULAR
       * -------------------------------------------------------------------
       * Nao e o layout de desktop encolhido: e outro layout. No celular o
       * painel deixa de ser uma janela flutuante de 380x460 no canto e passa
       * a ser uma folha ancorada embaixo, da largura da tela, com altura
       * limitada para nao engolir o exercicio.
       * ================================================================= */
      #${HOST_ID}.ka-mobile .ka-panel {
        left: 0; right: 0; bottom: 0; top: auto;
        width: 100%; max-width: none;
        /* 58vh: o painel cabe sem cobrir o enunciado, e o enunciado continua
           visivel acima. Em paisagem vira 78vh, porque sobra pouca altura. */
        height: 58vh; max-height: 78vh;
        border-radius: 14px 14px 0 0; border-left: none; border-right: none; border-bottom: none;
        box-shadow: 0 -12px 34px rgba(0,0,0,.5);
      }
      #${HOST_ID}.ka-mobile .ka-min { height: auto; }

      /* Cabecalho no celular: DUAS LINHAS.
         Antes o flex quebrava sozinho: o titulo virava duas linhas, o contador
         tres, e os botoes saiam cortados ("Inspec", "Config"). Ordenar com
         flex-order nao resolveu -- o flex-basis:100% do contador empurrava os
         botoes para uma terceira linha.
         Agora sao grupos explicitos no HTML:
           linha 1: estado + titulo + [ ] e X
           linha 2: contador + Iniciar, Inspecionar, Config */
      #${HOST_ID}.ka-mobile .ka-head {
        flex-wrap: wrap; row-gap: 8px; column-gap: 8px;
        padding: 11px 13px; cursor: grab;
      }
      #${HOST_ID}.ka-mobile .ka-head:active { cursor: grabbing; }
      /* No celular a ORDEM dos grupos e explicita.
         O grupo do contador tem flex-basis 100% (e o que forca a quebra de
         linha depois da linha 1). Se ele vier antes do grupo [ ] X no fluxo,
         ele consome a linha inteira e empurra os controles para uma TERCEIRA
         linha. Por isso: linha 1 = titulo + controles, linha 2 = contador +
         acoes. No desktop os grupos nem existem como itens flex
         (display:contents), entao order nao tem efeito la. */
      #${HOST_ID}.ka-mobile .ka-head > .ka-h1 { order: 1; }
      #${HOST_ID}.ka-mobile .ka-head > .ka-h3 { order: 2; }
      #${HOST_ID}.ka-mobile .ka-head > .ka-h2 { order: 3; }
      #${HOST_ID}.ka-mobile .ka-head > .ka-h1 {
        /* flex-basis 0, nao auto: com auto este grupo cresce ate ocupar tudo e
           empurra o [ ] X para uma terceira linha. Com 0 ele divide o espaco
           com o grupo dos controles de janela. */
        display: flex; align-items: center; gap: 8px; flex: 1 1 0; min-width: 0;
      }
      #${HOST_ID}.ka-mobile .ka-head > .ka-h2 {
        display: flex; align-items: center; gap: 8px; flex: 1 1 100%; min-width: 0;
      }
      #${HOST_ID}.ka-mobile .ka-head > .ka-h3 {
        display: flex; align-items: center; gap: 8px; flex: 0 0 auto;
      }
      #${HOST_ID}.ka-mobile .ka-title {
        /* Uma linha so, cortando com reticencias. min-width:0 e o que permite o
           ellipsis funcionar dentro de um flex item. */
        flex: 1 1 auto; min-width: 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        font-size: 15px;
      }
      #${HOST_ID}.ka-mobile .ka-stat {
        flex: 1 1 auto; min-width: 0;
        font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
      }
      /* As tres acoes dividem o espaco que sobra na linha 2. */
      #${HOST_ID}.ka-mobile #ka-run,
      #${HOST_ID}.ka-mobile #ka-inspect,
      #${HOST_ID}.ka-mobile #ka-cfg { flex: 1 1 0; min-width: 0; padding: 8px 6px; }
      #${HOST_ID}.ka-mobile #ka-inspect { font-size: 12.5px; }
      #${HOST_ID}.ka-mobile .ka-dot { width: 11px; height: 11px; }
      #${HOST_ID}.ka-mobile .ka-keydot { font-size: 13px; }

      #${HOST_ID}.ka-mobile .ka-btn {
        min-height: 38px; min-width: 38px; padding: 8px 12px;
        font-size: 13.5px; border-radius: 9px;
      }
      #${HOST_ID}.ka-mobile .ka-log { font-size: 13px; padding: 10px 12px; }
      #${HOST_ID}.ka-mobile .ka-input { padding: 10px; gap: 8px; }
      #${HOST_ID}.ka-mobile .ka-input textarea {
        /* 16px e o limiar em que o navegador deixa de dar zoom ao focar o
           campo. Abaixo disso a tela pula sozinha e atrapalha a digitacao. */
        font-size: 16px; min-height: 44px; padding: 10px 12px; border-radius: 10px;
      }
      #${HOST_ID}.ka-mobile .ka-cfg-scroll { padding: 14px; }
      #${HOST_ID}.ka-mobile .ka-lbl { font-size: 13px; margin: 14px 0 6px; }
      #${HOST_ID}.ka-mobile .ka-in { font-size: 16px; min-height: 42px; padding: 9px 11px; border-radius: 10px; }
      #${HOST_ID}.ka-mobile .ka-hint { font-size: 12px; }
      #${HOST_ID}.ka-mobile .ka-check { font-size: 14px; gap: 10px; margin-top: 16px; }
      #${HOST_ID}.ka-mobile .ka-check input { width: 20px; height: 20px; }
      #${HOST_ID}.ka-mobile .ka-cfg-foot { padding: 12px 14px; gap: 8px; }
      #${HOST_ID}.ka-mobile .ka-cfg-msg { font-size: 12.5px; }
      /* No celular a lista de modelos usa mais altura: rolar uma lista curta
         com o dedo e pior que rolar uma longa. */
      #${HOST_ID}.ka-mobile .ka-models { max-height: 190px; }
      #${HOST_ID}.ka-mobile .ka-model { font-size: 13px; padding: 9px 8px; }
      /* Duas colunas de campos numericos viram uma: em tela estreita os campos
         ficam ilegiveis lado a lado. */
      #${HOST_ID}.ka-mobile .ka-grid { grid-template-columns: 1fr; gap: 2px; }

      /* Paisagem: altura e o recurso escasso, entao o painel usa mais dela. */
      @media (orientation: landscape) and (max-height: 520px) {
        #${HOST_ID}.ka-mobile .ka-panel { height: 86vh; max-height: 92vh; }
      }
      /* Telas muito estreitas (320px): encolhe o texto em vez de quebrar de novo.
         A regra anterior forcava o titulo a 100% da largura, que era justamente
         o que produzia o cabecalho em quatro linhas. */
      @media (max-width: 380px) {
        #${HOST_ID}.ka-mobile .ka-title { font-size: 14px; }
        #${HOST_ID}.ka-mobile #ka-run,
        #${HOST_ID}.ka-mobile #ka-inspect,
        #${HOST_ID}.ka-mobile #ka-cfg { font-size: 12px; padding: 8px 4px; }
      }
      /* Toque sem ser celular (notebook com tela sensivel): alvos maiores, sem
         trocar o layout. */
      #${HOST_ID}.ka-touch.ka-desktop .ka-btn { min-height: 32px; padding: 6px 11px; }

      /* ===================================================================
       * PAINEL ARRASTADO (nos dois modos)
       * -------------------------------------------------------------------
       * O modo celular e uma folha ancorada embaixo: left/right 0 e 100% de
       * largura. Quando o usuario arrasta, essas amarras precisam sair, senao
       * o left/top que o arrasto escreve nao tem efeito visivel -- a folha
       * continuaria colada nas laterais.
       * O tamanho continua sendo o de celular (fonte grande, alvo grande); o
       * que muda e so a amarra de posicao.
       * ================================================================= */
      #${HOST_ID}.ka-mobile .ka-panel.ka-arrastado {
        left: auto; right: auto; bottom: auto; top: auto;
        width: min(340px, calc(100vw - 16px));
        height: 58vh; max-height: 70vh;
        border-radius: 14px; border: 1px solid #2a3040;
        box-shadow: 0 18px 48px rgba(0,0,0,.5);
      }
      #${HOST_ID}.ka-mobile .ka-panel.ka-arrastado.ka-min { height: auto; }
    `;
    (document.head || document.documentElement).appendChild(style);
  }

  function buildPanel() {
    if (document.getElementById(HOST_ID)) return;
    injectStyles();

    const host = document.createElement('div');
    host.id = HOST_ID;
    host.innerHTML = `
      <div class="ka-panel" id="ka-panel-el">
        <div class="ka-head" id="ka-head">
          <!-- Tres grupos que so existem no celular. No desktop, ".ka-head > div"
               tem display:contents, entao os filhos destes grupos entram no
               flex do cabecalho na MESMA ordem do HTML e o layout nao muda. -->
          <div class="ka-h1">
            <span class="ka-dot" id="ka-dot"></span>
            <span class="ka-title">Khan Agent</span>
            <span class="ka-keydot" id="ka-keydot" title="API key configurada">&#9679;</span>
          </div>
          <div class="ka-h2">
            <span class="ka-stat" id="ka-stat">0 acoes / 0 falhas</span>
            <button class="ka-btn go" id="ka-run">Iniciar</button>
            <button class="ka-btn" id="ka-inspect" title="Mostrar o que o script enxerga nesta pagina">Inspecionar</button>
            <button class="ka-btn" id="ka-cfg">Config</button>
          </div>
          <div class="ka-h3">
            <button class="ka-btn" id="ka-min">_</button>
            <button class="ka-btn" id="ka-x">X</button>
          </div>
        </div>
        <div class="ka-body">
          <div class="ka-log" id="ka-log"></div>

          <!-- ===== TELA DE CONFIGURACAO ===== -->
          <div class="ka-cfg" id="ka-cfg-view" hidden>
            <div class="ka-cfg-scroll">
              <label class="ka-lbl" for="ka-f-key">API key do Pollinations</label>
              <div class="ka-row">
                <input class="ka-in" id="ka-f-key" type="password" spellcheck="false"
                       placeholder="sk_... ou pk_... (enter.pollinations.ai/keys)">
                <button class="ka-btn" id="ka-f-eye" title="Mostrar/ocultar">ver</button>
              </div>
              <p class="ka-hint">Obrigatoria desde a deprecacao do modo anonimo. Fica no localStorage deste navegador.</p>

              <label class="ka-lbl" for="ka-f-backend">Backend</label>
              <select class="ka-in" id="ka-f-backend">
                <option value="http">HTTP direto (endpoint + API key)</option>
                <option value="puter">Puter (sem chave, conta do usuario)</option>
              </select>
              <div class="ka-puterbox" id="ka-puterbox">
                <p class="ka-hint" id="ka-puter-hint">Carregando o SDK do Puter...</p>
                <button type="button" class="ka-btn" id="ka-puter-signin">Entrar no Puter</button>
              </div>

              <label class="ka-lbl" for="ka-f-model">Modelo</label>
              <input class="ka-in" id="ka-f-model" type="text" spellcheck="false" list="ka-models" placeholder="openai">
              <datalist id="ka-models">
                <option value="openai"></option>
                <option value="openai-fast"></option>
                <option value="mistral"></option>
                <option value="qwen-coder"></option>
                <option value="gemini"></option>
                <option value="claude"></option>
                <option value="openai/gpt-oss-120b"></option>
                <option value="openai/gpt-oss-20b"></option>
                <option value="llama-3.3-70b-versatile"></option>
                <!-- Modelos do Puter conferidos contra o catalogo real (puter.ai.listModels),
                     filtrando por modalities.input === "image". deepseek-v4-pro
                     foi REMOVIDO de proposito: existe como texto, mas nao aceita
                     imagem, e sugerir modelo sem visao num agente que precisa
                     ver o grafico e apenas mandar o usuario a uma parede de erro. -->
                <option value="google/gemini-3.8-flash"></option>
                <option value="openai/gpt-5.6-luna"></option>
                <option value="anthropic/claude-opus-5"></option>
                <option value="qwen/qwen3.8-max"></option>
                <option value="qwen/qwen3-vl-plus"></option>
                <option value="qwen/qwen3-omni-flash"></option>
              </datalist>

              <!-- Catalogo vivo: busca os modelos no endpoint configurado.
                   O datalist acima continua servindo de atalho offline para os
                   ids conhecidos; a lista abaixo e o que o provider realmente
                   oferece agora, com filtro por texto. -->
              <div class="ka-row">
                <input class="ka-in" id="ka-f-modelsearch" type="search" spellcheck="false"
                       placeholder="buscar no catalogo (ex.: gemini, vl, :free)">
                <button class="ka-btn" id="ka-models-fetch" type="button"
                        title="Lista os modelos do endpoint configurado">Buscar</button>
              </div>
              <div class="ka-models" id="ka-models-list" hidden></div>

              <label class="ka-lbl" for="ka-f-endpoint">Endpoint</label>
              <input class="ka-in" id="ka-f-endpoint" type="text" spellcheck="false" list="ka-endpoints">
              <datalist id="ka-endpoints">
                <option value="https://gen.pollinations.ai/v1/chat/completions"></option>
                <option value="https://api.groq.com/openai/v1/chat/completions"></option>
                <option value="https://api.openai.com/v1/chat/completions"></option>
                <option value="https://openrouter.ai/api/v1/chat/completions"></option>
                <option value="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"></option>
                <option value="http://localhost:11434/v1/chat/completions"></option>
              </datalist>
              <p class="ka-hint" id="ka-provider-hint"></p>

              <div class="ka-grid">
                <div>
                  <label class="ka-lbl" for="ka-f-delay">Loop (ms)</label>
                  <input class="ka-in" id="ka-f-delay" type="number" min="500" step="100">
                </div>
                <div>
                  <label class="ka-lbl" for="ka-f-tries">Tentativas</label>
                  <input class="ka-in" id="ka-f-tries" type="number" min="1" max="10" step="1">
                </div>
                <div>
                  <label class="ka-lbl" for="ka-f-mem">Memoria</label>
                  <input class="ka-in" id="ka-f-mem" type="number" min="0" max="20" step="1">
                </div>
                <div>
                  <label class="ka-lbl" for="ka-f-timeout">Timeout (s)</label>
                  <input class="ka-in" id="ka-f-timeout" type="number" min="5" max="300" step="5">
                </div>
              </div>

              <label class="ka-check">
                <input type="checkbox" id="ka-f-autocheck"> Conferir a resposta automaticamente
              </label>
              <label class="ka-check">
                <input type="checkbox" id="ka-f-vision"> Enviar o grafico/imagem para a IA
              </label>
              <label class="ka-check">
                <input type="checkbox" id="ka-f-tools"> Usar tool calling nativo
              </label>
              <p class="ka-hint">Exige um modelo multimodal. Se o modelo recusar a imagem, o agente avisa e segue so com o texto. O tool calling tambem recua sozinho: se o provider recusar, volta ao JSON no texto.</p>
            </div>
            <div class="ka-cfg-foot">
              <span class="ka-cfg-msg" id="ka-cfg-msg"></span>
              <button class="ka-btn" id="ka-f-cancel">Fechar</button>
              <button class="ka-btn" id="ka-f-test">Testar</button>
              <button class="ka-btn go" id="ka-f-save">Salvar</button>
            </div>
          </div>

          <div class="ka-input">
            <textarea id="ka-msg" rows="1" placeholder="Fale com o assistente (Enter envia)"></textarea>
            <button class="ka-btn" id="ka-send">Enviar</button>
          </div>
        </div>
      </div>`;
    document.body.appendChild(host);

    ui = {
      host,
      panel: host.querySelector('#ka-panel-el'),
      head: host.querySelector('#ka-head'),
      dot: host.querySelector('#ka-dot'),
      stat: host.querySelector('#ka-stat'),
      run: host.querySelector('#ka-run'),
      inspectBtn: host.querySelector('#ka-inspect'),
      cfgBtn: host.querySelector('#ka-cfg'),
      min: host.querySelector('#ka-min'),
      x: host.querySelector('#ka-x'),
      log: host.querySelector('#ka-log'),
      msg: host.querySelector('#ka-msg'),
      send: host.querySelector('#ka-send'),
      // config
      cfgView: host.querySelector('#ka-cfg-view'),
      cfgMsg: host.querySelector('#ka-cfg-msg'),
      fKey: host.querySelector('#ka-f-key'),
      fEye: host.querySelector('#ka-f-eye'),
      fModel: host.querySelector('#ka-f-model'),
      fModelSearch: host.querySelector('#ka-f-modelsearch'),
      modelsFetch: host.querySelector('#ka-models-fetch'),
      modelsList: host.querySelector('#ka-models-list'),
      fEndpoint: host.querySelector('#ka-f-endpoint'),
      fBackend: host.querySelector('#ka-f-backend'),
      puterBox: host.querySelector('#ka-puterbox'),
      puterHint: host.querySelector('#ka-puter-hint'),
      puterSignIn: host.querySelector('#ka-puter-signin'),
      providerHint: host.querySelector('#ka-provider-hint'),
      fDelay: host.querySelector('#ka-f-delay'),
      fTries: host.querySelector('#ka-f-tries'),
      fMem: host.querySelector('#ka-f-mem'),
      fTimeout: host.querySelector('#ka-f-timeout'),
      fAutoCheck: host.querySelector('#ka-f-autocheck'),
      fVision: host.querySelector('#ka-f-vision'),
      fTools: host.querySelector('#ka-f-tools'),
      fSave: host.querySelector('#ka-f-save'),
      fTest: host.querySelector('#ka-f-test'),
      fCancel: host.querySelector('#ka-f-cancel'),
      keyDot: host.querySelector('#ka-keydot'),
    };

    ui.run.addEventListener('click', () => {
      state.running ? stopLoop() : startLoop();
    });
    ui.min.addEventListener('click', () => {
      ui.panel.classList.toggle('ka-min');
      ui.min.textContent = ui.panel.classList.contains('ka-min') ? '[ ]' : '_';
    });
    ui.x.addEventListener('click', () => { stopLoop(); ui.host.remove(); });
    ui.inspectBtn.addEventListener('click', () => {
      inspectPage().split('\n').forEach((l) => log(l, l.startsWith(' ') ? 'warn' : 'info'));
      showPanel(true);
    });
    ui.cfgBtn.addEventListener('click', () => showConfigView(ui.cfgView.hidden));
    ui.fCancel.addEventListener('click', () => showConfigView(false));
    ui.fSave.addEventListener('click', saveConfigFromForm);
    ui.fTest.addEventListener('click', testConfig);
    ui.fEye.addEventListener('click', () => {
      const show = ui.fKey.type === 'password';
      ui.fKey.type = show ? 'text' : 'password';
      ui.fEye.textContent = show ? 'ocul' : 'ver';
    });
    // A dica de provider acompanha o host digitado, para nao ter que adivinhar
    // o formato do model id de cada servico.
    ui.fEndpoint.addEventListener('input', updateProviderHint);
    ui.fEndpoint.addEventListener('change', updateProviderHint);

    // Catalogo de modelos: buscar no endpoint configurado e filtrar por texto.
    // O filtro e local (na lista ja baixada), entao digitar nao dispara rede.
    ui.modelsFetch.addEventListener('click', async () => {
      cfgMsg('Buscando modelos...', 'busy');
      try {
        const r = await buscarModelos({ forcar: true });
        state.modelos = r.lista;
        renderModelos();
        cfgMsg(`${r.lista.length} modelos em ${r.base}`, 'ok');
        log(`Catalogo: ${r.lista.length} modelos listados de ${r.base}.`);
      } catch (e) {
        cfgMsg(`Nao consegui listar: ${e.message}`, 'err');
        log(`Falha ao listar modelos: ${e.message}`, 'err');
      }
    });
    ui.fModelSearch.addEventListener('input', renderModelos);
    // Trocar de backend e o unico momento em que o SDK do Puter precisa
    // carregar, entao so carregamos quando ele realmente for usado.
    ui.fBackend.addEventListener('change', () => {
      paintBackend();
      if (ui.fBackend.value === 'puter') refreshPuterHint();
    });
    ui.puterSignIn.addEventListener('click', async () => {
      ui.puterSignIn.disabled = true;
      ui.puterHint.textContent = 'Abrindo a janela de login do Puter...';
      try {
        const quem = await puterEntrar();
        log(`Conectado ao Puter como ${quem}.`, 'ok');
        ui.puterHint.textContent = `Conectado como ${quem}. O consumo sai da sua conta Puter.`;
        ui.puterSignIn.textContent = 'Trocar de conta';
      } catch (e) {
        ui.puterHint.textContent = `Login falhou: ${e.message}`;
      } finally {
        ui.puterSignIn.disabled = false;
      }
    });
    // Enter dentro de um campo de texto salva; nos numeros tambem.
    ui.cfgView.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.matches('input:not([type=checkbox])')) {
        e.preventDefault();
        saveConfigFromForm();
      }
      if (e.key === 'Escape') showConfigView(false);
    });
    ui.send.addEventListener('click', sendChat);
    ui.msg.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
    });

    makeDraggable(ui.panel, ui.head);

    // Layout conforme o dispositivo, aplicado ja na criacao para nao piscar em
    // desktop e so depois virar folha.
    const disp = aplicarDispositivo();
    vigiarDispositivo();
    // No celular o painel comeca recolhido: 58vh de painel aberto por cima da
    // questao e o pior estado possivel, e o titulo ja mostra o estado do
    // agente. Um toque em [ ] abre.
    if (disp === 'mobile') {
      ui.panel.classList.add('ka-min');
      ui.min.textContent = '[ ]';
    }
  }

  // Arrasto por POINTER EVENTS, nao por mouse events.
  //
  // Com mousedown/mousemove o painel nao arrasta no celular: em toque, o
  // navegador emite touchstart -> touchend -> mousedown -> mouseup -> click, ou
  // seja o mousedown so chega DEPOIS que o dedo saiu. O resultado seria um
  // painel que aceita clique no cabecalho mas nunca acompanha o dedo.
  //
  // pointerdown/pointermove/pointerup unificam mouse, toque e caneta, e o
  // setPointerCapture garante que os eventos continuem chegando ao cabecalho
  // mesmo se o dedo sair da area dele no meio do gesto.
  function makeDraggable(panel, handle) {
    let sx = 0, sy = 0, ox = 0, oy = 0, dragging = false, idAtivo = null;

    const onDown = (e) => {
      if (e.target.closest('button')) return;
      // Botao direito/meio nao arrasta: no desktop isso abre o menu de contexto.
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      const r = panel.getBoundingClientRect();
      dragging = true;
      idAtivo = e.pointerId;
      // Marca que o usuario posicionou o painel a mao. No celular isso solta a
      // folha das laterais (ver .ka-arrastado no CSS): sem a marca, o left/top
      // escrito abaixo nao teria efeito, porque a folha esta ancorada em
      // left:0;right:0;bottom:0.
      panel.classList.add('ka-arrastado');
      sx = e.clientX; sy = e.clientY;
      ox = r.left; oy = r.top;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
      panel.style.left = `${r.left}px`;
      panel.style.top = `${r.top}px`;
      try { handle.setPointerCapture(e.pointerId); } catch { /* sem captura: segue com eventos globais */ }
      e.preventDefault();
    };

    const onMove = (e) => {
      if (!dragging || (idAtivo !== null && e.pointerId !== idAtivo)) return;
      const w = panel.offsetWidth, h = panel.offsetHeight;
      const x = Math.min(Math.max(0, ox + e.clientX - sx), Math.max(0, window.innerWidth - w));
      const y = Math.min(Math.max(0, oy + e.clientY - sy), Math.max(0, window.innerHeight - h));
      panel.style.left = `${x}px`;
      panel.style.top = `${y}px`;
      e.preventDefault();
    };

    const onUp = (e) => {
      if (!dragging) return;
      dragging = false;
      idAtivo = null;
      try { handle.releasePointerCapture(e.pointerId); } catch { /* ja solto */ }
    };

    handle.addEventListener('pointerdown', onDown);
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', onUp);
    handle.addEventListener('pointercancel', onUp);
    // Rede de seguranca: se o ponteiro for solto fora do cabecalho e a captura
    // falhar (navegador antigo), o arrasto ficaria preso ligado ao mouse.
    window.addEventListener('pointerup', onUp);
  }

  function paintStatus() {
    if (!ui.dot) return;
    ui.dot.className = 'ka-dot' + (state.busy ? ' busy' : state.running ? ' on' : '');
    ui.run.textContent = state.running ? 'Parar' : 'Iniciar';
    ui.run.className = 'ka-btn ' + (state.running ? 'stop' : 'go');
    ui.stat.textContent = `${state.solved} acoes / ${state.wrong} falhas`;
    // paintStatus sobrescreve o stat, entao o provider precisa ser reaplicado
    // aqui -- caso contrario o titulo volta a ficar vazio a cada observacao.
    ui.stat.title = `${providerName(state.cfg.endpoint)} · ${state.cfg.model}`;
    paintKeyDot();
  }

  // Nome curto do provider a partir do endpoint, para o painel mostrar com
  // clarity qual IA esta em uso. Sem isso, trocar de Groq para Pollinations (ou
  // vice-versa) deixavaduvida: so aparecia o nome do modelo, e a resposta vem
  // sempre no formato OpenAI, entao o modelo nao denuncia o host.
  function providerName(endpoint) {
    const e = String(endpoint || '');
    const m = e.match(/^https?:\/\/(?:api\.)?([^/]+)/i);
    const host = (m ? m[1] : e).toLowerCase();
    if (host.includes('pollinations')) return 'Pollinations';
    if (host.includes('groq')) return 'Groq';
    if (host.includes('openai.com')) return 'OpenAI';
    if (host.includes('together')) return 'Together';
    if (host.includes('openrouter')) return 'OpenRouter';
    if (host.includes('nvidia')) return 'NVIDIA NIM';
    if (host.includes('cohere')) return 'Cohere';
    if (host.includes('puter')) return 'Puter';
    if (host.includes('localhost') || host.includes('127.0.0.1')) return 'Local (Ollama/LM Studio)';
    if (host.includes('anthropic')) return 'Anthropic';
    return host || '(sem endpoint)';
  }

  // Bolinha ao lado do titulo: verde = key configurada, amarela = sem key.
  function paintKeyDot() {
    if (!ui.keyDot) return;
    const has = !!state.cfg.apiKey;
    ui.keyDot.className = 'ka-keydot' + (has ? ' on' : '');
    ui.keyDot.title = [
      `Provider: ${providerName(state.cfg.endpoint)}`,
      `Modelo: ${state.cfg.model}`,
      `Endpoint: ${state.cfg.endpoint}`,
      has ? `Key: ${maskKey(state.cfg.apiKey)}` : 'Sem API key',
      'Clique em Config para trocar',
    ].join('\n');

    // O botao Config carrega o provider+modelo, entao da para conferir qual IA
    // esta ativa sem abrir nada.
    if (ui.cfgBtn) {
      ui.cfgBtn.textContent = 'Config';
      ui.cfgBtn.title =
        `Provider: ${providerName(state.cfg.endpoint)}\n` +
        `Modelo: ${state.cfg.model}\n` +
        `Endpoint: ${state.cfg.endpoint}`;
    }
    if (ui.stat) {
      ui.stat.textContent = `${state.solved} acoes / ${state.wrong} falhas`;
      ui.stat.title = `${providerName(state.cfg.endpoint)} · ${state.cfg.model}`;
    }
  }

  function paintLog() {
    if (!ui.log) return;
    const atBottom = ui.log.scrollHeight - ui.log.scrollTop - ui.log.clientHeight < 40;
    ui.log.innerHTML = state.logLines
      .map((l) => `<div><span class="ka-t">${escapeHtml(l.at)}</span> <span class="ka-${l.kind}">${escapeHtml(l.msg)}</span></div>`)
      .join('');
    if (atBottom) ui.log.scrollTop = ui.log.scrollHeight;
  }

  function addChat(who, text) {
    log(`${who === 'user' ? 'voce' : 'assistente'}: ${text}`, who === 'user' ? 'info' : 'action');
  }

  // O que o agente esta vendo AGORA, em texto, para dar contexto ao chat.
  //
  // Sem isto o chat e um chatbot solto: o prompt dizia "assistente integrado a
  // um agente do Khan Academy", mas mandava so o texto digitado. O modelo nunca
  // recebia a tabela nem o enunciado, entao pedia ao usuario para confirmar
  // dados que estavam na tela -- e o usuario, com razao, lia isso como o
  // agente nao enxergar a pagina.
  //
  // Reaproveita takeSnapshot(), o mesmo extrator do loop, entao tudo o que
  // melhorou na extracao (MathML, tabelas) vale aqui de graca.
  function contextoDaPagina() {
    let snap = null;
    try { snap = takeSnapshot(); } catch { return { texto: '', imagem: null }; }
    if (!snap) return { texto: '', imagem: null };

    const l = [];
    l.push('CONTEXTO DA TELA (o agente esta vendo isto agora -- use estes dados');
    l.push('na resposta e NAO peca ao usuario para copiar o que ja esta aqui):');
    l.push(`URL: ${location.href}`);
    l.push('ENUNCIADO:');
    l.push(snap.problem);
    if (snap.choices.length) {
      l.push('');
      l.push(`ALTERNATIVAS (${snap.choices.length}), o indice e posicional:`);
      snap.choices.forEach((c, i) => l.push(`  [${i}] ${c.text}`));
    }
    if (snap.inputs.length) {
      l.push('');
      l.push(`CAMPOS DE RESPOSTA (${snap.inputs.length}):`);
      snap.inputs.forEach((f, i) => l.push(
        `  [${i}] ${f.kind === 'math' ? 'MATEMATICA' : 'texto'} `
        + `placeholder="${f.placeholder}" valor_atual="${f.value}"`
      ));
    }
    if (snap.progress) {
      l.push('');
      l.push(`PROGRESSO: ${snap.progress.done}/${snap.progress.total} `
        + `(${snap.progress.state}).`);
    }
    if (snap.feedback) {
      l.push(`ULTIMO FEEDBACK DA PLATAFORMA: ${snap.feedback}`);
    }

    // Se a questao tem figura e a visao esta ligada, o chat tambem enxerga:
    // "me explica esse grafico" e exatamente o pedido que isso atende.
    let imagem = null;
    if (snap.visual && state.cfg.vision && !state.visionDead) {
      imagem = snap.image || null; // decide() pode ja ter capturado
    }
    return { texto: l.join('\n'), imagem };
  }

  async function sendChat() {
    const text = ui.msg.value.trim();
    if (!text) return;
    ui.msg.value = '';
    addChat('user', text);
    state.history.push({ role: 'user', content: text });

    // O agente parou em ask_user esperando exatamente esta resposta.
    //
    // Antes o chat so conversava: o modelo resolvia a conta ali e NAO escrevia
    // nada na pagina, entao o usuario tinha de clicar em Iniciar de novo para o
    // agente agir -- e o sintoma relatado foi "ele so conversa e nao age".
    // Agora a resposta e o que faltava: volta para o loop, que ja le o historico
    // e portanto enxerga o que o usuario acabou de dizer.
    if (state.aguardandoUsuario && !state.running) {
      log('Resposta recebida. Retomando o agente com ela no contexto.', 'ok');
      startLoop();
      return;
    }

    try {
      const ctx = contextoDaPagina();
      const prompt = ctx.texto ? `${ctx.texto}\n\n---\n\nPERGUNTA DO USUARIO:\n${text}` : text;
      const reply = await askAI(
        'Voce e um assistente de estudo que acompanha o usuario no Khan Academy. '
        + 'Voce recebe o contexto da tela junto com a pergunta: use os numeros e o '
        + 'enunciado que estao nele, sem pedir ao usuario para copiar de novo. '
        + 'Responda em portugues, curto e direto, mostrando o calculo quando houver.',
        prompt,
        ctx.imagem
      );
      state.history.push({ role: 'assistant', content: reply });
      addChat('assistant', reply);
    } catch (err) {
      addChat('assistant', `Erro: ${err.message}`);
    }
  }

  /* ===========================================================================
   * SETTINGS
   * ======================================================================== */

  // Telas: 'log' mostra o log/chat, 'cfg' mostra o formulario de configuracao.
  //
  // showPanel(true) e o "tira do canto": usado por ask_user (a pergunta precisa
  // ficar visivel) e pelo botao Inspecionar (o dump vai para o log). O
  // recolhimento e a classe .ka-min, controlada pelo botao "_"; aqui a gente
  // so garante que o painel esteja aberto e no tamanho normal.
  function showPanel(show = true) {
    // O botao X faz ui.host.remove(): o nobo sai do DOM e some. Como o atalho do
    // bookmarklet existe para NAO reinjetar o arquivo (evita baixar 270 KB a cada
    // clique), ele precisa poder ressuscitar o painel. Sem esta volta, showPanel
    // mexia num nobo destacado, nao aparecia nada, e o favorito parecia quebrado.
    if (ui.host && !ui.host.isConnected) document.body.appendChild(ui.host);
    if (!ui.panel) return;
    const min = show ? ui.panel.classList.contains('ka-min') : false;
    if (show && min) {
      ui.panel.classList.remove('ka-min');
      if (ui.min) ui.min.textContent = '_';
    }
  }

  function showConfigView(show) {
    ui.cfgView.hidden = !show;
    ui.log.style.display = show ? 'none' : 'block';
    if (show) {
      fillConfigForm();
      carregarModelosDoCache();
      ui.fKey.focus();
    } else {
      cfgMsg('');
    }
  }

  // Dica de provider: o placeholder do campo de modelo muda conforme o host
  // digitado. Modelos de provider de terceiro tem prefixo proprio
  // ("openai/gpt-oss-120b" no Groq), e digitar so "openai" faz o Groq recusar
  // com 400 sem dizer qual era o id certo.
  const PROVIDER_HINT = [
    [/pollinations/i, 'Pollinations: model = "openai", "openai-fast". Key em enter.pollinations.ai/keys (sk_...).'],
    [/groq/i, 'Groq: model precisa do prefixo, ex.: "openai/gpt-oss-120b". Key gsk_... em console.groq.com.'],
    [/api\.openai\.com/i, 'OpenAI: model ex.: "gpt-4o-mini". Key sk-... em platform.openai.com.'],
    [/openrouter/i, 'OpenRouter: model ex.: "meta-llama/llama-3.3-70b-instruct". Key sk-or-...'],
    [/together/i, 'Together: model ex.: "meta-llama/Llama-3.3-70B-Instruct-Turbo".'],
    [/nvidia/i, 'NVIDIA NIM: a URL JA e o endpoint do modelo, sem "/chat/completions" (ex.: https://ai.api.nvidia.com/v1/vlm/google/paligemma). Exige a chave nvapi-... e exatamente UMA imagem por chamada.'],
    [/cohere/i, 'Cohere: a URL termina em /v2/chat, sem "/completions". Modelos com visao: "command-a-plus-05-2026". Chave cohere...'],
    [/generativelanguage\.googleapis\.com/i, 'Gemini: use o endpoint OpenAI-compat, terminando em /v1beta/openai/chat/completions, e model ex.: "gemini-2.5-flash". Key em aistudio.google.com/apikey. ATENCAO: o plano gratuito tem cota BAIXA (poucas chamadas por minuto) -- se aparecer 429, espere ~1 minuto.'],
    [/(localhost|127\.0\.0\.1)/i, 'Local: any OpenAI-compatible server (Ollama/LM Studio). Ex.: "llama3.2".'],
  ];

  function providerHintFor(endpoint) {
    const ep = endpoint || '';
    // O erro classico do Gemini: colar a URL NATIVA (a que termina em
    // ":generateContent"), que espera outro corpo. O sintoma e um 400 sem
    // relacao aparente com o problema, e a causa nao aparece em lugar nenhum.
    if (/generativelanguage\.googleapis\.com/i.test(ep) && /:generateContent|:streamGenerateContent/i.test(ep)) {
      return 'Essa e a URL NATIVA do Gemini, que usa outro formato de corpo. Troque para o endpoint '
        + 'OpenAI-compat: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';
    }
    for (const [re, txt] of PROVIDER_HINT) if (re.test(ep)) return txt;
    return 'Endpoint OpenAI-compatible: qualquer servidor que aceite POST /chat/completions.';
  }

  function updateProviderHint() {
    if (!ui.providerHint) return;
    const ep = ui.fEndpoint?.value || '';
    const partes = [];
    if (ep) partes.push(providerHintFor(ep));
    // Trocar de endpoint NAO troca o modelo, e o id costuma ser especifico do
    // provider ("openai" no Pollinations, "openai/gpt-oss-120b" no Groq).
    // Deixar isso implicito faz o usuario trocar e voltar com um modelo que o
    // provider novo nao conhece -- e o erro que aparece nao diz que a causa e
    // essa. O aviso e factual: nao adivinha, so lembra de conferir.
    const modelo = (ui.fModel?.value || '').trim();
    if (ep && modelo) {
      partes.push(`Modelo atual: "${modelo}" -- confirme com "Buscar" que ele existe neste provider.`);
    }
    ui.providerHint.textContent = partes.filter(Boolean).join(' ');
  }

  function fillConfigForm() {
    const c = state.cfg;
    ui.fKey.value = c.apiKey || '';
    ui.fModel.value = c.model || '';
    ui.fEndpoint.value = c.endpoint || '';
    ui.fDelay.value = c.loopDelayMs;
    ui.fTries.value = c.maxAttemptsPerQuestion;
    ui.fMem.value = c.memoryTurns;
    ui.fTimeout.value = Math.round(c.requestTimeoutMs / 1000);
    ui.fAutoCheck.checked = !!c.autoCheck;
    ui.fVision.checked = c.vision !== false;
    ui.fTools.checked = c.toolCalling !== false;
    // Trocar de provider/modelo reabre a conta: o "sem visao" marcado na
    // sessao anterior pode nao valer para o modelo novo.
    ui.fKey.type = 'password';
    ui.fEye.textContent = 'ver';
    ui.fBackend.value = c.backend === 'puter' ? 'puter' : 'http';
    paintBackend();
    updateProviderHint();
  }

  // No backend Puter nao existe endpoint nem chave: mostrar campos vazios
  // sugere que ainda falta configurar algo, e o usuario perde tempo procurando
  // uma key que nao existe.
  function paintBackend() {
    const puter = ui.fBackend && ui.fBackend.value === 'puter';
    if (ui.puterBox) ui.puterBox.style.display = puter ? '' : 'none';
    for (const el of [ui.fEndpoint, ui.fKey, ui.fEye]) {
      if (el) el.style.display = puter ? 'none' : '';
    }
    if (ui.providerHint) ui.providerHint.style.display = puter ? 'none' : '';
  }

  // Estado do SDK do Puter dentro do painel: se o script carregou, se a conta
  // esta conectada e qual provider/modelo esta em uso.
  async function refreshPuterHint() {
    if (!ui.puterHint) return;
    ui.puterHint.textContent = 'Carregando o SDK do Puter...';
    try {
      const p = await carregarPuter();
      const entrou = await puterAssinou();
      ui.puterHint.textContent = entrou
        ? 'SDK carregado e conta conectada. Modelo User-Pays: o consumo sai da sua conta Puter.'
        : 'SDK carregado. Clique em "Entrar no Puter" uma vez para conectar a conta.';
      if (ui.puterSignIn) {
        ui.puterSignIn.textContent = entrou ? 'Trocar de conta' : 'Entrar no Puter';
      }
      await listarModelosPuter();
    } catch (e) {
      ui.puterHint.textContent = `SDK indisponivel: ${e.message}`;
    }
  }

  // O catalogo do Puter tem ~1000 modelos, e o agente so se interessa pelos que
  // ACEITAM IMAGEM: sao eles que podem ver o grafico. Chutar nomes no datalist
  // era chute -- a lista abaixo vem da API, com o filtro certo.
  async function listarModelosPuter() {
    const dl = ui.fModel?.list;
    if (!dl) return;
    let ms = [];
    try {
      ms = (await window.puter.ai.listModels()) || [];
    } catch {
      return; // sem catalogo, o datalist estatico continua valendo
    }
    if (!Array.isArray(ms) || !ms.length) return;

    const aceitaImagem = (m) => Array.isArray(m?.modalities?.input) && m.modalities.input.includes('image');
    const visao = ms.filter(aceitaImagem);
    const preferidos = visao.filter((m) => /gemini|gpt-5|claude|qwen|llama/i.test(`${m.id} ${m.name}`));

    dl.innerHTML = '';
    const add = (m) => {
      const o = document.createElement('option');
      o.value = m.aliases?.[0] || m.puterId || m.id;
      dl.appendChild(o);
    };
    (preferidos.length ? preferidos : visao).slice(0, 120).forEach(add);
    if (!visao.length) {
      const o = document.createElement('option');
      o.value = 'google/gemini-3.8-flash';
      dl.appendChild(o);
    }

    if (ui.puterHint && visao.length) {
      const n = visao.length;
      ui.puterHint.textContent += ` ${n} dos ${ms.length} modelos aceitam imagem (lista preenchida).`;
    }
  }

  function cfgMsg(text, kind = '') {
    if (!ui.cfgMsg) return;
    ui.cfgMsg.textContent = text;
    ui.cfgMsg.className = 'ka-cfg-msg' + (kind ? ' ' + kind : '');
  }

  // Le o formulario -> state.cfg. Retorna false se algum numero for invalido.
  function readConfigForm() {
    const num = (el, lo, hi, fallback) => {
      const v = Number(el.value);
      if (!Number.isFinite(v)) return fallback;
      return Math.min(hi, Math.max(lo, Math.round(v)));
    };
    const model = ui.fModel.value.trim();
    if (!model) { cfgMsg('Informe um modelo.', 'err'); return false; }
    if (!/^[\w.\-/:]+$/.test(model)) {
      cfgMsg('Modelo invalido (use apenas letras, numeros, . - _ / :).', 'err');
      return false;
    }

    state.cfg.apiKey = ui.fKey.value.trim();
    state.cfg.model = model;
    const endpoint = ui.fEndpoint.value.trim() || CFG.endpoint;
    // Erro classico: colar a URL NATIVA do Gemini (a que termina em
    // ":generateContent"). Ela espera outro corpo, e o 400 que volta nao aponta
    // a causa em lugar nenhum -- o usuario so ve "nao aceita a API".
    if (/generativelanguage\.googleapis\.com/i.test(endpoint)
        && /:generateContent|:streamGenerateContent/i.test(endpoint)) {
      cfgMsg(
        'Essa e a URL NATIVA do Gemini, que usa outro formato de corpo. Use o endpoint '
        + 'OpenAI-compat: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
        'err'
      );
      return false;
    }
    state.cfg.endpoint = endpoint;
    state.cfg.loopDelayMs = num(ui.fDelay, 500, 30000, CFG.loopDelayMs);
    state.cfg.maxAttemptsPerQuestion = num(ui.fTries, 1, 10, CFG.maxAttemptsPerQuestion);
    state.cfg.memoryTurns = num(ui.fMem, 0, 20, CFG.memoryTurns);
    state.cfg.requestTimeoutMs = num(ui.fTimeout, 5, 300, 60) * 1000;
    state.cfg.backend = ui.fBackend.value === 'puter' ? 'puter' : 'http';
    state.cfg.autoCheck = ui.fAutoCheck.checked;
    state.cfg.vision = ui.fVision.checked;
    state.cfg.toolCalling = ui.fTools.checked;
    // Provider/modelo novo: nem o bloqueio de visao nem o formato de imagem
    // discoveries na sessao anterior valem para o endpoint novo.
    state.visionDead = false;
    state.imgStyle = null;
    state.singleMessage = false;
    // O suporte a tool calling e do MODELO, entao trocar de modelo reabre a
    // pergunta: um modelo que recusou nao diz nada sobre o proximo.
    state.toolsDead = false;
    state.toolChoiceDead = false;
    state.extrasDead = false;
    state.useGm = false;
    return true;
  }

  function saveConfigFromForm() {
    if (!readConfigForm()) return;
    saveCfg();
    paintKeyDot();
    cfgMsg('Salvo.', 'ok');
    log(
      `Config salva: ${providerName(state.cfg.endpoint)} model=${state.cfg.model} ` +
      `key=${state.cfg.apiKey ? maskKey(state.cfg.apiKey) : 'ausente'} ` +
      `loop=${state.cfg.loopDelayMs}ms tentativas=${state.cfg.maxAttemptsPerQuestion}`
    );
    log(`Endpoint em uso: ${state.cfg.endpoint}`);
  }

  function maskKey(k) {
    if (k.length <= 8) return '••••';
    return `${k.slice(0, 4)}…${k.slice(-2)}`;
  }

  /* ===========================================================================
   * CATALOGO DE MODELOS
   * ---------------------------------------------------------------------------
   * Deriva a URL de /models do proprio endpoint configurado, entao funciona em
   * qualquer provider OpenAI-compat sem uma tabela de hosts para manter.
   *
   * Capacidades: SO as que o provider declara. O OpenRouter devolve
   * architecture.input_modalities e supported_parameters, e ai os chips
   * "visao"/"tools" sao fato. Os outros endpoints devolvem so o id -- e nesse
   * caso o campo fica null (sem chip) em vez de um palpite: chip errado e pior
   * que chip ausente, porque leva o usuario a escolher um modelo que nao ve.
   * ======================================================================== */
  const MODELS_KEY = 'khan_agent_models_v1';
  const MODELS_TTL_MS = 12 * 3600e3;   // 12 h: pega modelo novo sem refazer sempre

  // "https://api.groq.com/openai/v1/chat/completions" -> ".../openai/v1"
  function baseDoEndpoint(endpoint) {
    return String(endpoint || '').trim().replace(/\/+$/, '').replace(/\/chat\/completions$/i, '');
  }

  function lerCacheModelos() {
    try {
      const raw = localStorage.getItem(MODELS_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }

  function gravarCacheModelos(v) {
    try { localStorage.setItem(MODELS_KEY, JSON.stringify(v)); } catch { /* storage cheio */ }
  }

  function normalizarModelos(data) {
    const brutos = Array.isArray(data?.data) ? data.data
      : Array.isArray(data?.models) ? data.models
        : Array.isArray(data) ? data : [];
    return brutos.map((m) => {
      // O fallback para `m` so vale quando o item E o id (string/numero). Com
      // um objeto sem id nem name, `String(m)` produzia "[object Object]" na
      // lista -- um modelo fantasma que nao existe para clicar.
      const bruto = m?.id ?? m?.name ?? (typeof m === 'string' || typeof m === 'number' ? m : '');
      const id = String(bruto).trim();
      if (!id) return null;
      const mods = m?.architecture?.input_modalities;
      const params = m?.supported_parameters;
      return {
        id,
        visao: Array.isArray(mods) ? mods.includes('image') : null,
        tools: Array.isArray(params) ? params.includes('tools') : null,
        ctx: Number(m?.context_window || m?.context_length || 0) || null,
      };
    }).filter(Boolean);
  }

  // forcar=true ignora o cache (botao "Buscar"); sem ele, o cache de 12 h serve
  // -- o catalogo da NVIDIA tem centenas de modelos e refazer a cada abertura
  // da Config e lento e desnecessario.
  async function buscarModelos({ forcar = false } = {}) {
    const base = baseDoEndpoint(state.cfg.endpoint);
    if (!base) throw new Error('o endpoint nao tem uma base para /models');

    const cache = lerCacheModelos();
    if (!forcar && cache && cache.base === base && Date.now() - cache.at < MODELS_TTL_MS) {
      return { ...cache, doCache: true };
    }

    const headers = {};
    if (state.cfg.apiKey) headers.Authorization = `Bearer ${state.cfg.apiKey}`;
    const res = await getJson(`${base}/models`, headers);
    if (!res.ok) {
      const det = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ao listar modelos${det ? `: ${det.slice(0, 120)}` : ''}`);
    }
    const data = await res.json();
    const lista = normalizarModelos(data);
    if (!lista.length) throw new Error('o endpoint respondeu, mas sem nenhum id de modelo');
    const saida = { base, at: Date.now(), lista };
    gravarCacheModelos(saida);
    return saida;
  }

  // Desenha a lista de modelos ja baixada, aplicando o filtro de texto.
  //
  // O corte em 200 e so de RENDERIZACAO: o catalogo da NVIDIA passa de 2000
  // ids e montar tudo em innerHTML trava a thread por um instante visivel. O
  // filtro continua valendo sobre a lista inteira, entao digitar acha qualquer
  // um -- o que nao cabe e apenas o que nao foi filtrado.
  function renderModelos() {
    if (!ui.modelsList) return;
    const lista = state.modelos || [];
    if (!lista.length) {
      ui.modelsList.hidden = true;
      return;
    }
    const filtro = (ui.fModelSearch.value || '').trim().toLowerCase();
    const achados = filtro ? lista.filter((m) => m.id.toLowerCase().includes(filtro)) : lista;
    const mostrados = achados.slice(0, 200);

    ui.modelsList.hidden = false;
    ui.modelsList.innerHTML =
      `<div class="ka-models-head">${achados.length} de ${lista.length} modelos`
      + (achados.length > mostrados.length ? ` (mostrando ${mostrados.length})` : '')
      + '</div>'
      + mostrados.map((m) => {
        // Chip so quando o provider DECLAROU a capacidade. Sem dado, sem chip.
        const chips = [];
        if (m.visao === true) chips.push('<span class="ka-chip visao">visao</span>');
        if (m.tools === true) chips.push('<span class="ka-chip tools">tools</span>');
        return `<button type="button" class="ka-model" data-id="${escapeHtml(m.id)}">`
          + `<span class="ka-model-id">${escapeHtml(m.id)}</span>${chips.join('')}</button>`;
      })
        .join('');

    for (const b of ui.modelsList.querySelectorAll('.ka-model')) {
      b.addEventListener('click', () => {
        ui.fModel.value = b.dataset.id;
        cfgMsg(`Modelo escolhido: ${b.dataset.id}`, 'ok');
      });
    }
  }

  // Ao abrir a Config, mostra o cache na hora (se houver) para nao exigir um
  // "Buscar" toda vez. O botao forca a atualizacao quando o usuario quiser.
  function carregarModelosDoCache() {
    const cache = lerCacheModelos();
    if (!cache || cache.base !== baseDoEndpoint(state.cfg.endpoint)) return;
    state.modelos = cache.lista || [];
    renderModelos();
  }

  // Requisicao minima para validar key + modelo + endpoint de uma vez.
  async function testConfig() {
    if (!readConfigForm()) return;
    cfgMsg('Testando...', 'busy');
    const started = Date.now();
    try {
      const reply = await askAI('Responda apenas com a palavra: ok', 'ping');
      const ms = Date.now() - started;
      const model = reply.slice(0, 40).replace(/\s+/g, ' ');
      cfgMsg(`OK em ${ms}ms - "${model}"`, 'ok');
      log(`Teste de conexao OK (${ms}ms) com ${state.cfg.model}.`);
    } catch (err) {
      const msg = err.needPuterLogin
        ? 'Puter: falta entrar na conta. Clique em "Entrar no Puter".'
        : err.auth
          ? (err.hint || 'Credencial recusada. Confira a key em Config.')
          : err.payment
            ? (err.hint || 'A conta do provider esta sem creditos (HTTP 402).')
            : err.message;
      cfgMsg(msg.slice(0, 200), 'err');
      log(`Teste de conexao falhou: ${err.message}`, 'err');
    }
  }

  /* ===========================================================================
   * NAVEGACAO / INIT
   * ======================================================================== */

  // Detecta troca de rota (SPA do Khan) para reiniciar o dedupe.
  function watchNavigation() {
    let last = location.href;
    setInterval(() => {
      if (location.href !== last) {
        last = location.href;
        state.epoch++;
        state.lastFingerprint = null;
        if (state.running) log('Navegou para outra pagina.', 'info');
      }
    }, 1000);
  }

  function init() {
    // Ja existe nesta pagina?
    //
    // Necessario para o modo BOOKMARKLET: o carregador e um <script src>, entao
    // clicar no favorito de novo injeta o arquivo OUTRA vez. Sem esta guarda,
    // existiriam dois paineis com o mesmo id, dois loops rodando e duas
    // contagens de acao. Com ela, o segundo clique so traz o painel de volta.
    if (window.__khanAgent && document.getElementById(HOST_ID)) {
      // `__khanAgentMostrar` e o atalho que o favorito chama. Evita um download
      // novo a cada clique.
      try { window.__khanAgentMostrar(); } catch { /* instalacao antiga */ }
      return;
    }
    if (window.__khanAgent) {
      // O painel foi fechado pelo botao X mas a instancia velha ficou solta no
      // gancho. Descarta antes de reconstruir, para o novo objeto nao ser
      // sobrescrito por outro com o mesmo nome.
      try { delete window.__khanAgent; } catch { /* pode ser propriedade so de leitura */ }
    }

    loadCfg();
    buildPanel();
    watchNavigation();
    paintStatus();

    // Gancho de inspecao. As funcoes do script vivem num IIFE e nenhuma delas
    // e alcançavel de fora, o que impossibilita testar caminho que nao passa
    // pela UI -- foi assim que o probe de visao e o de Puter enxergaram o que
    // enxergam. Exposto so para teste; o agente funciona sem ele.
    window.__khanAgent = {
      carregarPuter, puterAssinou, puterEntrar, puterPrompt, askPuter,
      dataUrlParaFile, extrairTexto, extrairNarracao, captureVisual, takeSnapshot,
      looksVisual, ehSvgDeGrafico, svgDoDomParaJpeg, screenshotDaArea,
      dicaDePagamento, reencodarRaster, fieldDeTextoVivo, valorNoCampoVivo,
      lerDica,
      buildUserPrompt, askAI, conferirChoiceText, normEscolha,
      calcular, TOOLS, toolSchemas, acoesDisponiveis, precondicaoDe, achaTool,
      performPlano, startLoop, stopLoop,
      buildMessages, partesDeImagem, capturarFiguraDe, capturarAlternativasSemTexto,
      svgToJpeg, escalaDeRender, VISION_MAX_PX, VISION_MIN_PX,
      digitarPorEventos, apagarPorEventos, mesmaExpressao, readMathField,
      criarEventoUI, humanClick,
      ACTIONS,
      state,
    };
    // Atalho chamado pelo bookmarklet quando o agente ja esta na pagina: mostra
    // o painel sem precisar reinjetar o arquivo.
    window.__khanAgentMostrar = () => showPanel(true);
    log('Painel pronto. Clique em Iniciar para o agente assumir.', 'ok');
    // O GM_xmlhttpRequest e o que permite falar com provider sem CORS. Dizer
    // na hora se ele existe evita a confusao classica: o agente funciona, a
    // imagem e capturada, e so na chamada da IA que aparece "Failed to fetch"
    // sem ninguem saber que a API faltou.
    log(
      gmAvailable()
        ? 'GM_xmlhttpRequest disponivel: providers sem CORS funcionam.'
        : 'GM_xmlhttpRequest AUSENTE: so funciona provider com CORS (Groq, OpenAI, '
          + 'OpenRouter, Cohere, Gemini, Mistral). Isso tem duas causas possiveis: '
          + 'carregado como BOOKMARKLET, que nao tem gerenciador de userscript '
          + '(use um provider com CORS); ou instalado no Tampermonkey sem recarregar '
          + 'a pagina do Khan depois de salvar.',
      gmAvailable() ? 'ok' : 'warn'
    );
    log(
      state.cfg.apiKey
        ? `Config carregada: ${providerName(state.cfg.endpoint)} model=${state.cfg.model} key=${maskKey(state.cfg.apiKey)}`
        : `Config: ${providerName(state.cfg.endpoint)} model=${state.cfg.model} SEM API key. Clique em Config para configurar.`,
      state.cfg.apiKey ? 'info' : 'warn'
    );

    // Atalhos: Alt+K liga/desliga o loop, Alt+C abre a configuracao.
    window.addEventListener('keydown', (e) => {
      if (!e.altKey) return;
      if (e.key === 'k' || e.key === 'K') {
        e.preventDefault();
        state.running ? stopLoop() : startLoop();
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        showConfigView(ui.cfgView.hidden);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
