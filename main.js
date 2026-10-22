/*
 * suminagashi — a living sheet of marbled paper.
 *
 * A GPU fluid simulation (stable fluids: semi-Lagrangian advection,
 * Jacobi pressure projection, vorticity confinement) drives three
 * pigments — indigo, madder, ochre — across warm fibrous paper.
 * Invisible hands stir the surface; the visitor stirs it too.
 */

const canvas = document.getElementById('c');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------------------------------------------------------------- config

const config = {
  SIM_RESOLUTION: 176,        // velocity grid (min dimension)
  DYE_RESOLUTION: 1152,       // pigment buffer (min dimension)
  PRESSURE_ITERATIONS: 24,
  PRESSURE_DECAY: 0.82,
  CURL: 18,                   // vorticity confinement — filigree within the flow
  VELOCITY_DISSIPATION: 1.8,  // high → viscous, meditative flow
  DYE_DISSIPATION: 0.022,     // strokes linger ~1min then return to paper
  DYE_SHARPEN: 0.03,          // unsharp mask — keeps filaments crisp
  SPLAT_FORCE: 5200,
};

// pigment channels: r = indigo, g = madder, b = ochre
const PIGMENTS = [
  [0.10, 0.14, 0.22],   // indigo (near-black blue)
  [0.55, 0.13, 0.09],   // madder red
  [0.78, 0.55, 0.16],   // ochre gold
];

// dye-channel basis: each pigment lives in one channel of the dye buffer
const CHANNELS = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

// ---------------------------------------------------------------- gl setup

const gl = canvas.getContext('webgl2', {
  alpha: false, depth: false, stencil: false, antialias: false,
  preserveDrawingBuffer: false,
});
if (!gl) {
  document.body.style.background = '#ece4d4';
  throw new Error('webgl2 unavailable');
}

const extCBF = gl.getExtension('EXT_color_buffer_float');
const extCBHF = extCBF ? null : gl.getExtension('EXT_color_buffer_half_float');
const supportLinearFloat = !!gl.getExtension('OES_texture_float_linear')
  || !!gl.getExtension('OES_texture_half_float_linear');
const halfFloatTexType = extCBF ? gl.HALF_FLOAT : (extCBHF ? gl.HALF_FLOAT : null);
const canRenderFloat = !!(extCBF || extCBHF);
const filterable = supportLinearFloat;

gl.clearColor(0, 0, 0, 1);

function compileShader(type, source, keywords) {
  if (keywords && keywords.length) {
    source = keywords.map(k => `#define ${k}\n`).join('') + source;
  }
  const s = gl.createShader(type);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

function createProgram(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.bindAttribLocation(p, 0, 'aPosition');
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(p));
  }
  const uniforms = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name;
    uniforms[name] = gl.getUniformLocation(p, name);
  }
  return { program: p, uniforms, bind() { gl.useProgram(p); } };
}

const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
  precision highp float;
  attribute vec2 aPosition;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform vec2 texelSize;
  void main () {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
  }
`);

const splatShader = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTarget;
  uniform float aspectRatio;
  uniform vec3 color;
  uniform vec2 point;
  uniform float radius;
  void main () {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    vec3 splat = exp(-dot(p, p) / radius) * color;
    vec3 base = texture2D(uTarget, vUv).xyz;
    gl_FragColor = vec4(base + splat, 1.0);
  }
`;

const vortexShader = `
  precision highp float;
  varying vec2 vUv;
  uniform sampler2D uTarget;
  uniform float aspectRatio;
  uniform vec2 point;
  uniform float radius;
  uniform float strength;
  void main () {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    float fall = exp(-dot(p, p) / radius);
    vec2 tang = vec2(-p.y, p.x) / (sqrt(dot(p, p)) + 0.02);
    vec2 v = texture2D(uTarget, vUv).xy + tang * strength * fall;
    gl_FragColor = vec4(v, 0.0, 1.0);
  }
`;

const advectionShader = `
  precision highp float;
  precision highp sampler2D;
  varying vec2 vUv;
  uniform sampler2D uVelocity;
  uniform sampler2D uSource;
  uniform vec2 texelSize;
  uniform vec2 dyeTexelSize;
  uniform float dt;
  uniform float dissipation;
  vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {
    vec2 st = uv / tsize - 0.5;
    vec2 iuv = floor(st);
    vec2 fuv = fract(st);
    vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);
    vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);
    vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);
    vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);
    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
  }
  void main () {
  #ifdef MANUAL_FILTERING
    vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;
    vec4 result = bilerp(uSource, coord, dyeTexelSize);
  #else
    vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
    vec4 result = texture2D(uSource, coord);
  #endif
    float decay = 1.0 + dissipation * dt;
    gl_FragColor = result / decay;
  }
`;

const divergenceShader = `
  precision mediump float;
  precision mediump sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).x;
    float R = texture2D(uVelocity, vR).x;
    float T = texture2D(uVelocity, vT).y;
    float B = texture2D(uVelocity, vB).y;
    vec2 C = texture2D(uVelocity, vUv).xy;
    if (vL.x < 0.0) { L = -C.x; }
    if (vR.x > 1.0) { R = -C.x; }
    if (vT.y > 1.0) { T = -C.y; }
    if (vB.y < 0.0) { B = -C.y; }
    float div = 0.5 * (R - L + T - B);
    gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
  }
`;

const curlShader = `
  precision mediump float;
  precision mediump sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uVelocity, vL).y;
    float R = texture2D(uVelocity, vR).y;
    float T = texture2D(uVelocity, vT).x;
    float B = texture2D(uVelocity, vB).x;
    float vorticity = R - L - T + B;
    gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
  }
`;

const vorticityShader = `
  precision highp float;
  precision highp sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uVelocity;
  uniform sampler2D uCurl;
  uniform float curl;
  uniform float dt;
  void main () {
    float L = texture2D(uCurl, vL).x;
    float R = texture2D(uCurl, vR).x;
    float T = texture2D(uCurl, vT).x;
    float B = texture2D(uCurl, vB).x;
    float C = texture2D(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity += force * dt;
    velocity = min(max(velocity, -1000.0), 1000.0);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`;

const pressureShader = `
  precision mediump float;
  precision mediump sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uPressure;
  uniform sampler2D uDivergence;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    float divergence = texture2D(uDivergence, vUv).x;
    float pressure = (L + R + B + T - divergence) * 0.25;
    gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
  }
`;

const sharpenShader = `
  precision highp float;
  precision highp sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uTexture;
  uniform float amount;
  void main () {
    vec3 c = texture2D(uTexture, vUv).rgb;
    vec3 blur = 0.25 * (texture2D(uTexture, vL).rgb + texture2D(uTexture, vR).rgb
                      + texture2D(uTexture, vT).rgb + texture2D(uTexture, vB).rgb);
    vec3 r = c + (c - blur) * amount;
    // pigment never reaches the deckle — keeps borders clean
    float m = smoothstep(0.0, 0.035, vUv.x) * smoothstep(0.0, 0.035, 1.0 - vUv.x)
            * smoothstep(0.0, 0.035, vUv.y) * smoothstep(0.0, 0.035, 1.0 - vUv.y);
    gl_FragColor = vec4(clamp(r * m, 0.0, 8.0), 1.0);
  }
`;

const clearShader = `
  precision mediump float;
  precision mediump sampler2D;
  varying vec2 vUv;
  uniform sampler2D uTexture;
  uniform float value;
  void main () {
    gl_FragColor = value * texture2D(uTexture, vUv);
  }
`;

const gradientSubtractShader = `
  precision mediump float;
  precision mediump sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uPressure;
  uniform sampler2D uVelocity;
  void main () {
    float L = texture2D(uPressure, vL).x;
    float R = texture2D(uPressure, vR).x;
    float T = texture2D(uPressure, vT).x;
    float B = texture2D(uPressure, vB).x;
    vec2 velocity = texture2D(uVelocity, vUv).xy;
    velocity.xy -= vec2(R - L, T - B);
    gl_FragColor = vec4(velocity, 0.0, 1.0);
  }
`;

// the artist — maps pigment channels to ink on paper
const displayShader = `
  precision highp float;
  precision highp sampler2D;
  varying vec2 vUv;
  varying vec2 vL;
  varying vec2 vR;
  varying vec2 vT;
  varying vec2 vB;
  uniform sampler2D uDye;
  uniform vec2 dyeTexelSize;
  uniform float uTime;
  uniform float uAspect;

  float hash (vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }
  float noise (vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
      mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
  }
  float fbm (vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
      v += a * noise(p);
      p = p * 2.03 + vec2(17.3, 9.1);
      a *= 0.5;
    }
    return v;
  }

  void main () {
    vec3 dye = texture2D(uDye, vUv).rgb;

    // ---- paper: warm ivory with mottling, fibres, grain
    vec2 puv = vUv * vec2(uAspect, 1.0);
    float mottle = fbm(puv * 3.0);
    float fibreH = noise(vec2(puv.x * 220.0, puv.y * 9.0));
    float fibreV = noise(vec2(puv.x * 9.0, puv.y * 220.0));
    float grain  = hash(vUv * 913.7) - 0.5;
    vec3 paper = vec3(0.937, 0.908, 0.839);
    paper *= 0.965 + 0.07 * mottle;
    paper *= 1.0 - 0.020 * fibreH - 0.016 * fibreV;
    paper += grain * 0.018;

    // ---- pigment
    vec3 indigo = vec3(${PIGMENTS[0].join(',')});
    vec3 madder = vec3(${PIGMENTS[1].join(',')});
    vec3 ochre  = vec3(${PIGMENTS[2].join(',')});
    float wI = 1.0 - exp(-dye.r * 1.35);
    float wM = 1.0 - exp(-dye.g * 1.55);
    float wO = 1.0 - exp(-dye.b * 1.75);
    float total = wI + wM + wO;
    vec3 pig = (indigo * wI + madder * wM + ochre * wO) / max(total, 1e-4);
    pig *= 0.94 + 0.12 * fbm(puv * 6.0);   // no ink is ever quite even
    float amount = 1.0 - exp(-total * 1.35);

    // crisp printed contours — paper shows between filaments
    amount = smoothstep(0.09, 0.52, amount);
    vec3 col = mix(paper, pig * (0.92 + 0.10 * mottle), amount * 0.95);

    // ---- relief: ink sits on the paper, lit from upper left
    vec3 dL = texture2D(uDye, vL).rgb;
    vec3 dR = texture2D(uDye, vR).rgb;
    vec3 dT = texture2D(uDye, vT).rgb;
    vec3 dB = texture2D(uDye, vB).rgb;
    float sL = dL.r + dL.g + dL.b;
    float sR = dR.r + dR.g + dR.b;
    float sT = dT.r + dT.g + dT.b;
    float sB = dB.r + dB.g + dB.b;
    vec2 grad = vec2(sR - sL, sT - sB);
    float relief = dot(grad, normalize(vec2(-0.6, 0.75)));
    col += relief * 0.020 * amount;

    // filament outlines — marbled ink dries with a dark edge
    float edge = clamp(length(grad) * 0.4, 0.0, 0.20);
    col *= 1.0 - edge * 0.5;

    // ---- ochre glints: sparse metallic flecks in gold passages
    float fleck = step(0.9965, hash(floor(vUv * vec2(uAspect, 1.0) * 700.0)));
    float shimmer = 0.5 + 0.5 * sin(uTime * 0.6 + hash(vUv * 311.0) * 40.0);
    col += fleck * wO * shimmer * vec3(0.30, 0.22, 0.06) * amount;

    // ---- vignette: paper recedes into the dark, under a drifting lamp
    vec2 lamp = vec2(0.5 + 0.14 * sin(uTime * 0.043 + 1.7),
                     0.52 + 0.10 * sin(uTime * 0.031));
    vec2 v = vUv - lamp;
    v.x *= uAspect;
    float vig = smoothstep(0.95, 0.30, length(v));
    col *= 0.78 + 0.22 * vig;
    col *= 1.0 + 0.05 * vig;

    gl_FragColor = vec4(col, 1.0);
  }
`;

const progSplat      = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, splatShader));
const progVortex     = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, vortexShader));
const progAdvection  = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, advectionShader,
  filterable ? null : ['MANUAL_FILTERING']));
const progDivergence = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, divergenceShader));
const progCurl       = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, curlShader));
const progVorticity  = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, vorticityShader));
const progPressure   = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, pressureShader));
const progSharpen    = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, sharpenShader));
const progClear      = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, clearShader));
const progGradSub    = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, gradientSubtractShader));
const progDisplay    = createProgram(baseVertexShader, compileShader(gl.FRAGMENT_SHADER, displayShader));

// fullscreen quad
const quadVAO = gl.createVertexArray();
gl.bindVertexArray(quadVAO);
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

function blit(target) {
  if (target == null) {
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  } else {
    gl.viewport(0, 0, target.width, target.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
  }
  gl.bindVertexArray(quadVAO);
  gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
  gl.bindVertexArray(null);
}

// ---------------------------------------------------------------- fbos

function createFBO(w, h, internalFormat, format, type, filter) {
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return {
    texture, fbo, width: w, height: h,
    texelSizeX: 1 / w, texelSizeY: 1 / h,
    attach(id) {
      gl.activeTexture(gl.TEXTURE0 + id);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      return id;
    },
  };
}

function createDoubleFBO(w, h, internalFormat, format, type, filter) {
  let fbo1 = createFBO(w, h, internalFormat, format, type, filter);
  let fbo2 = createFBO(w, h, internalFormat, format, type, filter);
  return {
    width: w, height: h,
    texelSizeX: 1 / w, texelSizeY: 1 / h,
    get read() { return fbo1; },
    set read(v) { fbo1 = v; },
    get write() { return fbo2; },
    set write(v) { fbo2 = v; },
    swap() { const t = fbo1; fbo1 = fbo2; fbo2 = t; },
  };
}

function getResolution(base) {
  let aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
  if (aspect < 1) aspect = 1 / aspect;
  const min = Math.round(base);
  const max = Math.round(base * aspect);
  return gl.drawingBufferWidth > gl.drawingBufferHeight
    ? { width: max, height: min }
    : { width: min, height: max };
}

let dye, velocity, divergence, curl, pressure;

function initFramebuffers() {
  const simRes = getResolution(config.SIM_RESOLUTION);
  const dyeRes = getResolution(config.DYE_RESOLUTION);
  const texType = canRenderFloat ? halfFloatTexType : gl.UNSIGNED_BYTE;
  const internal = extCBF ? gl.RGBA16F : (extCBHF ? gl.RGBA16F : gl.RGBA);
  const filtering = (canRenderFloat && filterable) ? gl.LINEAR : gl.NEAREST;

  if (dye) { /* drop old textures implicitly via GC */ }
  dye = createDoubleFBO(dyeRes.width, dyeRes.height, internal, gl.RGBA, texType, filtering);
  velocity = createDoubleFBO(simRes.width, simRes.height, internal, gl.RGBA, texType, filtering);
  divergence = createFBO(simRes.width, simRes.height, internal, gl.RGBA, texType, gl.NEAREST);
  curl = createFBO(simRes.width, simRes.height, internal, gl.RGBA, texType, gl.NEAREST);
  pressure = createDoubleFBO(simRes.width, simRes.height, internal, gl.RGBA, texType, gl.NEAREST);
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    return true;
  }
  return false;
}

resizeCanvas();
initFramebuffers();

// ---------------------------------------------------------------- splats

function correctRadius(radius) {
  const aspect = canvas.width / canvas.height;
  if (aspect > 1) radius *= aspect;
  return radius;
}

function stir(x, y, dx, dy, radius) {
  progSplat.bind();
  gl.uniform1i(progSplat.uniforms.uTarget, velocity.read.attach(0));
  gl.uniform1f(progSplat.uniforms.aspectRatio, canvas.width / canvas.height);
  gl.uniform2f(progSplat.uniforms.point, x, y);
  gl.uniform1f(progSplat.uniforms.radius, correctRadius(radius));
  gl.uniform3f(progSplat.uniforms.color, dx, dy, 0);
  blit(velocity.write);
  velocity.swap();
}

function drop(x, y, channel, amount, radius, burst) {
  progSplat.bind();
  gl.uniform1i(progSplat.uniforms.uTarget, dye.read.attach(0));
  gl.uniform1f(progSplat.uniforms.aspectRatio, canvas.width / canvas.height);
  gl.uniform2f(progSplat.uniforms.point, x, y);
  const c = CHANNELS[channel];
  gl.uniform3f(progSplat.uniforms.color,
    c[0] * amount, c[1] * amount, c[2] * amount);
  gl.uniform1f(progSplat.uniforms.radius, correctRadius(radius));
  blit(dye.write);
  dye.swap();

  if (burst) {
    // radial push — a bloom, not a blob
    const n = 8;
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + Math.random() * 0.4;
      const px = x + Math.cos(a) * radius * 0.5;
      const py = y + Math.sin(a) * radius * 0.5;
      const f = burst * (0.7 + Math.random() * 0.6);
      stir(px, py, Math.cos(a) * f, Math.sin(a) * f, radius * 0.8);
    }
  }
}

function vortex(x, y, strength, radius) {
  progVortex.bind();
  gl.uniform1i(progVortex.uniforms.uTarget, velocity.read.attach(0));
  gl.uniform1f(progVortex.uniforms.aspectRatio, canvas.width / canvas.height);
  gl.uniform2f(progVortex.uniforms.point, x, y);
  gl.uniform1f(progVortex.uniforms.radius, correctRadius(radius));
  gl.uniform1f(progVortex.uniforms.strength, strength);
  blit(velocity.write);
  velocity.swap();
}

// ---------------------------------------------------------------- choreography
// invisible hands: wanderers that lay pigment, scheduled gestures.

const rand = (a, b) => a + Math.random() * (b - a);
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

let simTime = 0;
let events = [];          // { t, kind, ... } one-shot gestures
let hands = [];           // pigment-laying emitters, always at work
let comb = null;          // active comb pass
let strokes = [];         // traveling calligraphic gestures
let energy = 1;           // smoothed vigor of the invisible hands
let energyTarget = 1;
let phaseT = 0;
let phaseDur = rand(8, 12);
let nextDrop = 6, nextVortex = 15, nextComb = 55, nextStroke = 12;

// pigments appear in eras — indigo leads, madder answers, ochre accents
const PIGMENT_WEIGHTS = [0.42, 0.33, 0.25];
function pickChannel() {
  const r = Math.random();
  return r < PIGMENT_WEIGHTS[0] ? 0 : r < PIGMENT_WEIGHTS[0] + PIGMENT_WEIGHTS[1] ? 1 : 2;
}

// each hand tends its own quarter of the sheet — presences stay distinct
const ZONES = [
  [0.26, 0.42], [0.50, 0.60], [0.72, 0.44],
  [0.36, 0.66], [0.62, 0.34], [0.48, 0.42],
];

function newHand(channel, zoneIdx) {
  const cx = ZONES[zoneIdx][0] + rand(-0.06, 0.06);
  const cy = ZONES[zoneIdx][1] + rand(-0.06, 0.06);
  return {
    channel, zone: zoneIdx,
    x: cx, y: cy,
    // each hand follows a slow lissajous-like wander around its territory
    a: rand(0, Math.PI * 2), b: rand(0, Math.PI * 2),
    wa: rand(0.05, 0.11), wb: rand(0.05, 0.11),
    ra: rand(0.06, 0.14), rb: rand(0.05, 0.12),
    cx, cy,
    dyeRate: rand(0.012, 0.032),
    force: rand(260, 460),
    life: rand(14, 26),
    age: 0,
  };
}

function freeZone() {
  const used = new Set(hands.map(h => h.zone));
  const free = ZONES.map((_, i) => i).filter(i => !used.has(i));
  return free.length ? pick(free) : Math.floor(rand(0, ZONES.length));
}

function scheduleDrop(delay) {
  events.push({
    t: simTime + delay, kind: 'drop',
    x: rand(0.18, 0.82), y: rand(0.2, 0.8),
    channel: pickChannel(),
    amount: rand(0.25, 0.55),
    radius: rand(0.006, 0.02),
    burst: rand(60, 200) * (Math.random() < 0.3 ? 1 : 0),
  });
}

function scheduleVortex(delay) {
  events.push({
    t: simTime + delay, kind: 'vortex',
    x: rand(0.25, 0.75), y: rand(0.3, 0.7),
    strength: pick([-1, 1]) * rand(2.2, 4.5),
    radius: rand(0.05, 0.11),
  });
}

function scheduleComb(delay) {
  // a rake dragged across the sheet — the signature ebru gesture
  events.push({
    t: simTime + delay, kind: 'comb',
    horizontal: Math.random() < 0.75,
    dir: pick([-1, 1]),
    wobble: rand(0.03, 0.07),
    duration: rand(2.8, 3.8),
    teeth: Math.floor(rand(7, 11)),
    force: rand(170, 320),
  });
}

function scheduleStroke(delay) {
  // a single sinuous brush-line drawn clear across the sheet
  const flip = Math.random() < 0.5;
  events.push({
    t: simTime + delay, kind: 'stroke',
    p0: [flip ? rand(0.68, 0.86) : rand(0.10, 0.30), rand(0.2, 0.8)],
    p1: [rand(0.3, 0.7), rand(0.08, 0.92)],
    p2: [flip ? rand(0.10, 0.30) : rand(0.68, 0.86), rand(0.2, 0.8)],
    channel: pickChannel(),
    dur: rand(2.6, 4.2),
    width: rand(0.0025, 0.006),
    amount: rand(0.05, 0.11),
  });
}

function emitHand(h, dt) {
  const px = h.cx + Math.sin(h.a) * h.ra;
  const py = h.cy + Math.cos(h.b) * h.rb;
  const dx = (px - h.x), dy = (py - h.y);
  h.x = px; h.y = py;
  // hand stirs and trails pigment
  stir(h.x, h.y, dx * h.force * (0.4 + 0.6 * energy), dy * h.force * (0.4 + 0.6 * energy), 0.016);
  progSplat.bind();
  gl.uniform1i(progSplat.uniforms.uTarget, dye.read.attach(0));
  gl.uniform1f(progSplat.uniforms.aspectRatio, canvas.width / canvas.height);
  gl.uniform2f(progSplat.uniforms.point, h.x, h.y);
  const c = CHANNELS[h.channel];
  const fade = Math.min(1, (h.life - h.age) / 4) * Math.min(1, h.age / 3);
  const rate = h.dyeRate * fade * energy;
  gl.uniform3f(progSplat.uniforms.color,
    c[0] * rate, c[1] * rate, c[2] * rate);
  gl.uniform1f(progSplat.uniforms.radius, correctRadius(rand(0.002, 0.005)));
  blit(dye.write);
  dye.swap();
  h.a += h.wa * dt * 60 * 0.16;
  h.b += h.wb * dt * 60 * 0.16;
}

function updateChoreography(dt) {
  // the breath of the piece — periods of vigor alternating with stillness
  phaseT += dt;
  if (phaseT > phaseDur) {
    phaseT = 0;
    if (energyTarget > 0.6) {
      energyTarget = 0.22;
      phaseDur = rand(16, 26);
      if (Math.random() < 0.55) scheduleComb(rand(1, 4));
    } else {
      energyTarget = 1.0;
      phaseDur = rand(10, 18);
      if (Math.random() < 0.5) scheduleVortex(rand(2, 6));
    }
  }
  energy += (energyTarget - energy) * Math.min(1, dt * 0.8);

  for (let i = hands.length - 1; i >= 0; i--) {
    const h = hands[i];
    h.age += dt;
    if (h.age > h.life) { hands.splice(i, 1); continue; }
    emitHand(h, dt);
  }
  while (hands.length < 3) {
    // keep all three pigments at work — a missing channel gets the next brush
    const missing = [0, 1, 2].filter(c => !hands.some(h => h.channel === c));
    hands.push(newHand(missing.length ? pick(missing) : pickChannel(), freeZone()));
  }

  // comb pass
  if (comb) {
    comb.t += dt;
    const k = comb.t / comb.duration;
    if (k >= 1) { comb = null; }
    else {
      for (let i = 0; i < comb.teeth; i++) {
        const u = (i + 0.5) / comb.teeth;
        let x, y, dx, dy;
        if (comb.horizontal) {
          x = comb.dir > 0 ? k : 1 - k;
          y = 0.08 + u * 0.84 + Math.sin(k * 9 + i) * comb.wobble;
          dx = comb.dir; dy = 0;
        } else {
          y = comb.dir > 0 ? k : 1 - k;
          x = 0.08 + u * 0.84 + Math.sin(k * 9 + i) * comb.wobble;
          dx = 0; dy = comb.dir;
        }
        stir(x, y, dx * comb.force, dy * comb.force, 0.012);
      }
    }
  }

  // brush strokes — a hand swept along a bezier, trailing ink
  for (let i = strokes.length - 1; i >= 0; i--) {
    const s = strokes[i];
    s.t += dt;
    const k = Math.min(1, s.t / s.dur);
    const u = 1 - k;
    const x = u * u * s.p0[0] + 2 * u * k * s.p1[0] + k * k * s.p2[0];
    const y = u * u * s.p0[1] + 2 * u * k * s.p1[1] + k * k * s.p2[1];
    if (s.px !== undefined) {
      stir(x, y, (x - s.px) * 900, (y - s.py) * 900, 0.010);
      progSplat.bind();
      gl.uniform1i(progSplat.uniforms.uTarget, dye.read.attach(0));
      gl.uniform1f(progSplat.uniforms.aspectRatio, canvas.width / canvas.height);
      gl.uniform2f(progSplat.uniforms.point, x, y);
      const c = CHANNELS[s.channel];
      const taper = Math.sin(Math.PI * k);           // brush lands and lifts
      gl.uniform3f(progSplat.uniforms.color,
        c[0] * s.amount * taper, c[1] * s.amount * taper, c[2] * s.amount * taper);
      gl.uniform1f(progSplat.uniforms.radius,
        correctRadius(s.width * (0.55 + 0.45 * taper)));
      blit(dye.write);
      dye.swap();
    }
    s.px = x; s.py = y;
    if (k >= 1) strokes.splice(i, 1);
  }

  // scheduled one-shots
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (simTime < e.t) continue;
    events.splice(i, 1);
    if (e.kind === 'drop') drop(e.x, e.y, e.channel, e.amount, e.radius, e.burst);
    else if (e.kind === 'vortex') vortex(e.x, e.y, e.strength, e.radius);
    else if (e.kind === 'comb' && !comb) comb = { ...e, t: 0 };
    else if (e.kind === 'stroke') strokes.push({ ...e, t: 0 });
  }

  // the slow calendar — drops fall, eddies turn, the rake passes
  if (simTime > nextDrop) { scheduleDrop(0); nextDrop = simTime + rand(9, 20); }
  if (simTime > nextVortex) { scheduleVortex(0); nextVortex = simTime + rand(20, 45); }
  if (simTime > nextComb) { scheduleComb(0); nextComb = simTime + rand(70, 120); }
  if (simTime > nextStroke) { scheduleStroke(0); nextStroke = simTime + rand(18, 40); }
}

// opening composition — a print already mid-life
function prime() {
  // a few ancestral drops, combed once — the print begins mid-gesture
  drop(0.34, 0.56, 0, 0.9, 0.018, 40);
  drop(0.62, 0.46, 1, 0.7, 0.015, 30);
  drop(0.50, 0.62, 2, 0.6, 0.013, 0);
  drop(0.44, 0.42, 0, 0.5, 0.012, 0);
  vortex(0.52, 0.52, 2.6, 0.07);
  for (let i = 0; i < 140; i++) step(1 / 60);
}

// ---------------------------------------------------------------- pointer

const pointer = { x: 0.5, y: 0.5, dx: 0, dy: 0, down: false, moved: false, channel: 0 };
let pointerColorTimer = 0;

function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / r.width,
    y: 1 - (e.clientY - r.top) / r.height,
  };
}

canvas.addEventListener('pointermove', e => {
  const p = canvasPos(e);
  pointer.dx += (p.x - pointer.x);
  pointer.dy += (p.y - pointer.y);
  pointer.x = p.x; pointer.y = p.y;
  pointer.moved = true;
});
canvas.addEventListener('pointerdown', e => {
  const p = canvasPos(e);
  pointer.x = p.x; pointer.y = p.y;
  pointer.down = true;
  // a drop of whatever pigment the brush currently carries
  drop(p.x, p.y, pointer.channel, rand(0.7, 1.2), rand(0.02, 0.04), rand(200, 500));
});
window.addEventListener('pointerup', () => { pointer.down = false; });

function applyPointer(dt) {
  if (!pointer.moved) return;
  pointer.moved = false;
  const speed = Math.hypot(pointer.dx, pointer.dy);
  if (speed < 0.0002) { pointer.dx = pointer.dy = 0; return; }
  // stir the paint; drag a faint trail of the carried pigment
  stir(pointer.x, pointer.y,
    pointer.dx * config.SPLAT_FORCE, pointer.dy * config.SPLAT_FORCE,
    0.018);
  progSplat.bind();
  gl.uniform1i(progSplat.uniforms.uTarget, dye.read.attach(0));
  gl.uniform1f(progSplat.uniforms.aspectRatio, canvas.width / canvas.height);
  gl.uniform2f(progSplat.uniforms.point, pointer.x, pointer.y);
  const c = CHANNELS[pointer.channel];
  const amt = Math.min(0.20, speed * 3.0);
  gl.uniform3f(progSplat.uniforms.color, c[0] * amt, c[1] * amt, c[2] * amt);
  gl.uniform1f(progSplat.uniforms.radius, correctRadius(0.010));
  blit(dye.write);
  dye.swap();
  pointer.dx = pointer.dy = 0;
}

// ---------------------------------------------------------------- sim step

function step(dt) {
  gl.disable(gl.BLEND);

  progCurl.bind();
  gl.uniform2f(progCurl.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  gl.uniform1i(progCurl.uniforms.uVelocity, velocity.read.attach(0));
  blit(curl);

  progVorticity.bind();
  gl.uniform2f(progVorticity.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  gl.uniform1i(progVorticity.uniforms.uVelocity, velocity.read.attach(0));
  gl.uniform1i(progVorticity.uniforms.uCurl, curl.attach(1));
  gl.uniform1f(progVorticity.uniforms.curl, config.CURL);
  gl.uniform1f(progVorticity.uniforms.dt, dt);
  blit(velocity.write);
  velocity.swap();

  progDivergence.bind();
  gl.uniform2f(progDivergence.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  gl.uniform1i(progDivergence.uniforms.uVelocity, velocity.read.attach(0));
  blit(divergence);

  progClear.bind();
  gl.uniform2f(progClear.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  gl.uniform1i(progClear.uniforms.uTexture, pressure.read.attach(0));
  gl.uniform1f(progClear.uniforms.value, config.PRESSURE_DECAY);
  blit(pressure.write);
  pressure.swap();

  progPressure.bind();
  gl.uniform2f(progPressure.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  gl.uniform1i(progPressure.uniforms.uDivergence, divergence.attach(0));
  for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
    gl.uniform1i(progPressure.uniforms.uPressure, pressure.read.attach(1));
    blit(pressure.write);
    pressure.swap();
  }

  progGradSub.bind();
  gl.uniform2f(progGradSub.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  gl.uniform1i(progGradSub.uniforms.uPressure, pressure.read.attach(0));
  gl.uniform1i(progGradSub.uniforms.uVelocity, velocity.read.attach(1));
  blit(velocity.write);
  velocity.swap();

  progAdvection.bind();
  gl.uniform2f(progAdvection.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
  if (!filterable)
    gl.uniform2f(progAdvection.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
  const velId = velocity.read.attach(0);
  gl.uniform1i(progAdvection.uniforms.uVelocity, velId);
  gl.uniform1i(progAdvection.uniforms.uSource, velId);
  gl.uniform1f(progAdvection.uniforms.dt, dt);
  gl.uniform1f(progAdvection.uniforms.dissipation, config.VELOCITY_DISSIPATION);
  blit(velocity.write);
  velocity.swap();

  if (!filterable)
    gl.uniform2f(progAdvection.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
  gl.uniform1i(progAdvection.uniforms.uVelocity, velocity.read.attach(0));
  gl.uniform1i(progAdvection.uniforms.uSource, dye.read.attach(1));
  gl.uniform1f(progAdvection.uniforms.dissipation, config.DYE_DISSIPATION);
  blit(dye.write);
  dye.swap();

  // unsharp mask — pushes back against advection diffusion, keeps filaments alive
  progSharpen.bind();
  gl.uniform2f(progSharpen.uniforms.texelSize, dye.texelSizeX, dye.texelSizeY);
  gl.uniform1i(progSharpen.uniforms.uTexture, dye.read.attach(0));
  gl.uniform1f(progSharpen.uniforms.amount, config.DYE_SHARPEN);
  blit(dye.write);
  dye.swap();
}

function render(time) {
  progDisplay.bind();
  gl.uniform2f(progDisplay.uniforms.texelSize, dye.texelSizeX, dye.texelSizeY);
  gl.uniform2f(progDisplay.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
  gl.uniform1i(progDisplay.uniforms.uDye, dye.read.attach(0));
  gl.uniform1f(progDisplay.uniforms.uTime, time);
  gl.uniform1f(progDisplay.uniforms.uAspect, canvas.width / canvas.height);
  blit(null);
}

// ---------------------------------------------------------------- run

prime();

let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  simTime += dt;

  // the brush slowly changes pigment as it travels
  pointerColorTimer += dt;
  if (pointerColorTimer > 22) {
    pointerColorTimer = 0;
    pointer.channel = (pointer.channel + 1) % 3;
  }

  if (resizeCanvas()) { initFramebuffers(); prime(); }
  updateChoreography(dt);
  applyPointer(dt);
  step(dt);
  render(now / 1000);
  requestAnimationFrame(frame);
}

if (reducedMotion) {
  // a still print: run the sim a while longer, then hold
  for (let i = 0; i < 240; i++) step(1 / 60);
  render(0);
  // keep the canvas responsive to resizes, but still
  window.addEventListener('resize', () => {
    if (resizeCanvas()) { initFramebuffers(); prime(); for (let i = 0; i < 240; i++) step(1 / 60); }
    render(0);
  });
} else {
  requestAnimationFrame(frame);
}
