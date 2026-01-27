// NOTE: These standard node imports are often used in custom routers for file system or path logic.
// They are commented out here for simplicity in a self-contained example.
// const path = require('node:path');
// const os = require('node:os');
// const fs = require('node:fs/promises');

// --- CONSTANTS ---
// High-performance coding models
const GLM_FLASH = 'zai,glm-4.5-flash';
const GLM_PREMIUM = 'zai,glm-4.7';

// Context & Background models
const GEMINI_LIGHT = 'gemini,gemini-2.0-flash-exp';
const GEMINI_FLASH_LATEST = 'gemini,gemini-flash-latest';
const GEMINI_PRO = 'gemini,gemini-pro-latest'; // Long Context Champion

// Specialized Provider Defaults
const KIMI_K2_THINKING = 'moonshot,kimi-k2-thinking-turbo';
const PERPLEXITY_SONAR = 'perplexity,sonar-pro';
const FALLBACK_BACKUP = 'deepseek,deepseek-chat';

// --- STATEFUL CIRCUIT BREAKER ---
// NOTE: This stateful logic requires the router to be loaded as a singleton, typically in the server.
const failureTracker = new Map();
const COOLDOWN_MS = 60000;

function extractText(content) {
  if (!content) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.filter(b => b.type === 'text').map(b => b.text).join('\n');
  }
  return String(content);
}

module.exports = {
  // Handler for when a request fails due to a quota or rate limit error
  onRequestError(routeKey, error) {
    const msg = error?.message || '';
    // Standardize error detection for quotas/limits
    if (msg.includes('429') || msg.includes('quota') || msg.includes('limit') || msg.includes('concurrency')) {
      console.log(`[Router] 🛑 Quota hit for ${routeKey}. Pausing for 60s.`);
      failureTracker.set(routeKey, Date.now() + COOLDOWN_MS);
    }
  },

  /**
   * Main routing function.
   * @param {Object} req - The incoming chat request object.
   * @param {Object} config - The merged configuration object (including Router settings).
   * @returns {string} The provider,model string (e.g., 'anthropic,claude-3-haiku').
   */
  async route(req, config) {
    const messages = req.body?.messages || [];
    const lastUserMsgObj = messages.filter(m => m.role === 'user').pop();
    const prompt = extractText(lastUserMsgObj?.content);

    // CONFIG: Get routes from config.json or use defaults
    const ROUTE_RESEARCH = config.Router?.webSearch || PERPLEXITY_SONAR;
    const ROUTE_THINK = config.Router?.think || KIMI_K2_THINKING;
    const ROUTE_LONG_CTX = config.Router?.longContext || GEMINI_PRO;
    const ROUTE_DEFAULT = config.Router?.default || GLM_FLASH;
    const ROUTE_BACKGROUND = config.Router?.background || GEMINI_LIGHT;

    // STATE: Is this a build session? (Look at history for [BUILD] tag)
    const isBuildSession = messages.some(m => /\[BUILD\]/i.test(extractText(m.content)));

    // STATE: Is the premium Z.ai model currently in cooldown?
    const isGlmPremiumLimited = failureTracker.has(GLM_PREMIUM) && Date.now() < failureTracker.get(GLM_PREMIUM);

    let routeModel = '';
    let tag = 'STANDARD';

    // --- 1. SPECIALIZED AGENT ROUTING (Top Priority) ---

    // A. Research & Web Search -> Perplexity
    // Catches [RESEARCH-COMPLEXITY], [RESEARCH], or explicit search tags
    if (/\[(RESEARCH|SEARCH)\]/i.test(prompt) || prompt.includes("RESEARCH-COMPLEXITY")) {
      console.log(`[Router] 🔍 Routing to Research Provider (Perplexity)`);
      return ROUTE_RESEARCH;
    }

    // B. Thinking & Planning -> Moonshot (Kimi k2)
    // Catches [PLAN], [SPEC-CHECK], [THINK]
    if (/\[(PLAN|SPEC-CHECK|THINK)\]/i.test(prompt)) {
      console.log(`[Router] 🧠 Routing to Reasoning Provider (Moonshot)`);
      return ROUTE_THINK;
    }

    // C. Long Context & Architecture -> Gemini Pro [NEW SECTION]
    // Catches: [ARCHITECT], [MAP], or natural language cues
    if (/\[(ARCHITECT|DESIGN|MAP|ANALYSIS|CODEBASE)\]/i.test(prompt) ||
        /\b(cartographer|map out|analyze codebase)\b/i.test(prompt)) {
      console.log(`[Router] 🗺️ Routing to Long Context Provider (Gemini Pro)`);
      return ROUTE_LONG_CTX;
    }

    // --- 2. HEAVY LIFTING (BUILD/CODE) ---

    // D. Build Tags -> GLM Premium
    if (/\[(BUILD)\]/i.test(prompt)) {
      routeModel = GLM_PREMIUM;
      tag = 'BUILD_TASK';

      // Failover for Build Tasks
      if (isGlmPremiumLimited) {
        console.log(`[Router] 🛡️ GLM-4.7 limited. Failing over to DeepSeek for Build.`);
        routeModel = FALLBACK_BACKUP;
        tag = 'BUILD_FAILOVER';
      }
    }

    if (routeModel) return routeModel;

    // E. General Coding Keywords -> GLM Premium
    const codingKeywords = /\b(code|function|class|debug|implement|refactor|typescript|python|javascript)\b/i;
    if (codingKeywords.test(prompt)) {
      routeModel = GLM_PREMIUM;
      // Failover for Coding Tasks
      if (isGlmPremiumLimited) {
        routeModel = FALLBACK_BACKUP;
      }
      return routeModel;
    }

    // --- 3. THE "QUOTA PROTECTOR" DEFAULT LOGIC ---
    /* LOGIC:
       1. If we are in a BUILD SESSION (history has [BUILD]), we use Gemini for Haiku tasks immediately.
       2. If GLM_PREMIUM has hit a quota, we assume the Z.ai account is pressured and shift Haiku to Gemini.
       3. Otherwise, use GLM_FLASH (Z.ai) as the primary reflex.
    */

    const isZaiUnderPressure = isBuildSession || isGlmPremiumLimited;

    if (isZaiUnderPressure) {
      // Use Background/Light model (Gemini)
      routeModel = ROUTE_BACKGROUND;
      tag = isBuildSession ? 'HAIKU_PROTECT_BUILD' : 'HAIKU_ZAI_PRESSURE';

      // Secondary check: if Gemini Light is also limited, fallback to Gemini Flash Latest
      const isGeminiLightLimited = failureTracker.has(GEMINI_LIGHT) && Date.now() < failureTracker.get(GEMINI_LIGHT);
      if (isGeminiLightLimited) {
        routeModel = GEMINI_FLASH_LATEST;
      }
    } else {
      // Default Reflex (Z.ai Flash)
      routeModel = ROUTE_DEFAULT;
      tag = 'HAIKU_REFLEX';
    }

    // --- 4. FINAL FAILOVER CHAIN ---
    // If the selected model is currently ratelimited, walk down the chain
    const isLimited = (m) => failureTracker.has(m) && Date.now() < failureTracker.get(m);

    if (isLimited(routeModel)) {
      // If Z.ai Flash is limited -> Go to Gemini
      if (routeModel.includes('zai')) {
        routeModel = GEMINI_LIGHT;
      }
      // If Gemini is limited -> Go to DeepSeek
      else if (routeModel.includes('gemini')) {
        routeModel = FALLBACK_BACKUP;
      }
    }

    console.log(`[Router] ${tag} → ${routeModel}`);
    return routeModel;
  }
};
