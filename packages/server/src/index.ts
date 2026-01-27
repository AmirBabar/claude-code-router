import { existsSync, readFileSync } from "fs";
import { writeFile } from "fs/promises";
import { homedir } from "os";
import { join } from "path";
import { initConfig, initDir } from "./utils";
import { createServer } from "./server";
import { apiKeyAuth } from "./middleware/auth";
import { CONFIG_FILE, HOME_DIR, listPresets } from "@CCR/shared";
import { createStream } from 'rotating-file-stream';
import { sessionUsageCache } from "@musistudio/llms";
import { SSEParserTransform } from "./utils/SSEParser.transform";
import { SSESerializerTransform } from "./utils/SSESerializer.transform";
import { rewriteStream } from "./utils/rewriteStream";
import JSON5 from "json5";
import { IAgent, ITool } from "./agents/type";
import agentsManager from "./agents";
import { EventEmitter } from "node:events";
import { pluginManager, tokenSpeedPlugin } from "@musistudio/llms";
import { PassThrough, Readable } from "stream"; // [CCR_HOOK] Import PassThrough and Readable
import { CustomRouter } from "./types/router"; // [CCR_HOOK] Import CustomRouter types
// [FIX] Use default import for transformerRegistry to resolve TS2614
// [FIX] Use default import for the registry
import transformerRegistry from "@musistudio/llms";

// [FIX] Extract the class from the registry object instead of a named import
// This bypasses the "no exported member" error because we know it exists at runtime on the default object
const { PerplexityTransformer } = transformerRegistry as any;

// [FIX] Register it
(transformerRegistry as any)["perplexity"] = PerplexityTransformer;

const event = new EventEmitter()

async function initializeClaudeConfig() {
  const homeDir = homedir();
  const configPath = join(homeDir, ".claude.json");
  if (!existsSync(configPath)) {
    const userID = Array.from(
      { length: 64 },
      () => Math.random().toString(16)[2]
    ).join("");
    const configContent = {
      numStartups: 184,
      autoUpdaterStatus: "enabled",
      userID,
      hasCompletedOnboarding: true,
      lastOnboardingVersion: "1.0.17",
      projects: {},
    };
    await writeFile(configPath, JSON.stringify(configContent, null, 2));
  }
}

interface RunOptions {
  port?: number;
  logger?: any;
}

/**
 * Plugin configuration from config file
 */
interface PluginConfig {
  name: string;
  enabled?: boolean;
  options?: Record<string, any>;
}

/**
 * Register plugins from configuration
 * @param serverInstance Server instance
 * @param config Application configuration
 */
async function registerPluginsFromConfig(serverInstance: any, config: any): Promise<void> {
  // Get plugins configuration from config file
  const pluginsConfig: PluginConfig[] = config.plugins || config.Plugins || [];

  for (const pluginConfig of pluginsConfig) {
      const { name, enabled = false, options = {} } = pluginConfig;

      switch (name) {
        case 'token-speed':
          pluginManager.registerPlugin(tokenSpeedPlugin, {
            enabled,
            outputHandlers: [
              {
                type: 'temp-file',
                enabled: true
              }
            ],
            ...options
          });
          break;

        default:
          console.warn(`Unknown plugin: ${name}`);
          break;
      }
    }
  // Enable all registered plugins
  await pluginManager.enablePlugins(serverInstance);
}

async function getServer(options: RunOptions = {}) {
  await initializeClaudeConfig();
  await initDir();
  const config = await initConfig();

  // Check if Providers is configured
  const providers = config.Providers || config.providers || [];
  const hasProviders = providers && providers.length > 0;

  let HOST = config.HOST || "127.0.0.1";

  if (hasProviders) {
    HOST = config.HOST;
    if (!config.APIKEY) {
      HOST = "127.0.0.1";
    }
  } else {
    // When no providers are configured, listen on 0.0.0.0 without authentication
    HOST = "0.0.0.0";
    console.log("ℹ️  No providers configured. Listening on 0.0.0.0 without authentication.");
  }

  const port = config.PORT || 3456;

  // Use port from environment variable if set (for background process)
  const servicePort = process.env.SERVICE_PORT
    ? parseInt(process.env.SERVICE_PORT)
    : port;

  // Configure logger based on config settings or external options
  const pad = (num: number) => (num > 9 ? "" : "0") + num;
  const generator = (time: number | Date | undefined, index: number | undefined) => {
    let date: Date;
    if (!time) {
      date = new Date();
    } else if (typeof time === 'number') {
      date = new Date(time);
    } else {
      date = time;
    }

    const month = date.getFullYear() + "" + pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hour = pad(date.getHours());
    const minute = pad(date.getMinutes());

    return `./logs/ccr-${month}${day}${hour}${minute}${pad(date.getSeconds())}${index ? `_${index}` : ''}.log`;
  };

  let loggerConfig: any;

  // Use external logger configuration if provided
  if (options.logger !== undefined) {
    loggerConfig = options.logger;
  } else {
    // Enable logger if not provided and config.LOG !== false
    if (config.LOG !== false) {
      // Set config.LOG to true (if not already set)
      if (config.LOG === undefined) {
        config.LOG = true;
      }
      loggerConfig = {
        level: config.LOG_LEVEL || "debug",
        stream: createStream(generator, {
          path: HOME_DIR,
          maxFiles: 3,
          interval: "1d",
          compress: false,
          maxSize: "50M"
        }),
      };
    } else {
      loggerConfig = false;
    }
  }

  const presets = await listPresets();

  const serverInstance = await createServer({
    jsonPath: CONFIG_FILE,
    initialConfig: {
      // ...config,
      providers: config.Providers || config.providers,
      HOST: HOST,
      PORT: servicePort,
      LOG_FILE: join(
        homedir(),
        ".claude-code-router",
        "claude-code-router.log"
      ),
    },
    logger: loggerConfig,
  });

  await Promise.allSettled(
      presets.map(async preset => await serverInstance.registerNamespace(`/preset/${preset.name}`, preset.config))
  )

  // Register and configure plugins from config
  await registerPluginsFromConfig(serverInstance, config);

  // Add async preHandler hook for authentication
  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    return new Promise<void>((resolve, reject) => {
      const done = (err?: Error) => {
        if (err) reject(err);
        else resolve();
      };
      // Call the async auth function
      apiKeyAuth(config)(req, reply, done).catch(reject);
    });
  });
  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    const url = new URL(`http://127.0.0.1${req.url}`);
    req.pathname = url.pathname;
    if (req.pathname.endsWith("/v1/messages") && req.pathname !== "/v1/messages") {
      req.preset = req.pathname.replace("/v1/messages", "").replace("/", "");
    }
  });

  // [CCR_HOOK] Add preHandler hook for custom router logic
  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    if (req.pathname.endsWith("/v1/messages") && config.CUSTOM_ROUTER_PATH && existsSync(config.CUSTOM_ROUTER_PATH)) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        let customRouter: CustomRouter = require(config.CUSTOM_ROUTER_PATH);

        // Backward compatibility: wrap function-based routers
        if (typeof customRouter === "function") {
          customRouter = { route: customRouter };
        }

        // Store router for other hooks
        req.customRouter = customRouter;

        // Get routing decision
        let routeKey = await customRouter.route(req, config);

        // Check if router needs to queue this request
        if (customRouter.canAcquireSlot) {
          const acquired = await customRouter.canAcquireSlot(routeKey);
          if (!acquired) {
            // Queue timeout - get failover route
            routeKey = await customRouter.route(req, { ...config, failover: true });
          }
        }

        req.routeKey = routeKey; // Store for onSend/onError hooks

        // [CCR_HOOK] CRITICAL: Override the model in the request body to enforce the custom route.
        if (req.body && typeof req.body === 'object') {
          req.body.model = routeKey;
        }

        // Notify router that request is starting
        if (customRouter.onRequestStart) {
          customRouter.onRequestStart(req.routeKey);
        }
      } catch (err: any) {
        req.log.error(`[CCR Hook Error] Custom router preHandler failed: ${err.message}`);
        // Do not crash the server, proceed with default routing
      }
    }
  });

  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    if (req.pathname.endsWith("/v1/messages")) {
      const useAgents = []

      for (const agent of agentsManager.getAllAgents()) {
        if (agent.shouldHandle(req, config)) {
          // Set agent identifier
          useAgents.push(agent.name)

          // change request body
          agent.reqHandler(req, config);

          // append agent tools
          if (agent.tools.size) {
            if (!req.body?.tools?.length) {
              req.body.tools = []
            }
            req.body.tools.unshift(...Array.from(agent.tools.values()).map(item => {
              return {
                name: item.name,
                description: item.description,
                input_schema: item.input_schema
              }
            }))
          }
        }
      }

      if (useAgents.length) {
        req.agents = useAgents;
      }
    }
  });
  serverInstance.addHook("onError", async (request: any, reply: any, error: any) => {
    // [CCR_HOOK] Add onError hook logic
    if (request.customRouter && request.routeKey) {
      try {
        if (request.customRouter.onRequestError) {
          request.customRouter.onRequestError(request.routeKey, error);
        }
      } catch (err: any) {
        request.log.error(`[CCR Hook Error] Custom router onError failed: ${err.message}`);
      }
    }
    event.emit('onError', request, reply, error);
  })
  serverInstance.addHook("onSend", (req: any, reply: any, payload: any, done: any) => {
      // @ts-ignore - custom properties
  const { customRouter, routeKey } = req;

  if (customRouter && routeKey) {
    // Case 1: Streaming Response
    if (payload && (payload instanceof Readable || typeof (payload as any).pipe === "function")) {
      const stream = payload as Readable;
      let completed = false;
      const onComplete = () => {
        if (!completed) {
          completed = true;
          try {
            customRouter.onRequestComplete && customRouter.onRequestComplete(routeKey, req, reply);
          } catch (err) {
            req.log.error(`[CCR Hook Error] Custom router onRequestComplete failed: ${(err as Error).message}`);
          }
        }
      };

      // Hook into stream events
      stream.on("end", onComplete);
      stream.on("finish", onComplete);
      stream.on("error", () => {
        // Error handling is usually done in onRequestError, but ensure we don't leak
        if (!completed) onComplete();
      });
    }
    // Case 2: Non-Streaming (Buffered) Response (THE FIX)
    else {
      try {
        // Call complete immediately for buffered responses
        customRouter.onRequestComplete && customRouter.onRequestComplete(routeKey, req, reply);
      } catch (err) {
        req.log.error(`[CCR Hook Error] Custom router onRequestComplete (buffered) failed: ${(err as Error).message}`);
      }
    }
  }

  done(null, payload);
  });
  serverInstance.addHook("onSend", async (req: any, reply: any, payload: any) => {
    event.emit('onSend', req, reply, payload);
    return payload;
  });

  // Add global error handlers to prevent the service from crashing
  process.on("uncaughtException", (err) => {
    serverInstance.app.log.error("Uncaught exception:", err);
  });

  process.on("unhandledRejection", (reason, promise) => {
    serverInstance.app.log.error("Unhandled rejection at:", promise, "reason:", reason);
  });

  return serverInstance;
}

async function run() {
  const server = await getServer();
  server.app.post("/api/restart", async () => {
    setTimeout(async () => {
      process.exit(0);
    }, 100);

    return { success: true, message: "Service restart initiated" }
  });

  // Dashboard route
  server.app.get("/ui/dashboard", async (req: any, reply: any) => {
    try {
      const htmlPath = join(__dirname, "ui", "dashboard.html");
      const html = readFileSync(htmlPath, "utf8");

      reply.header("Content-Type", "text/html; charset=utf-8");
      reply.header("Cache-Control", "no-cache");
      return reply.send(html);
    } catch (error: any) {
      console.error("[Server] Failed to serve dashboard:", error);
      return reply.code(500).send("Dashboard unavailable");
    }
  });

  await server.start();
}

export { getServer };
export type { RunOptions };
export type { IAgent, ITool } from "./agents/type";
export { initDir, initConfig, readConfigFile, writeConfigFile, backupConfigFile } from "./utils";
export { pluginManager, tokenSpeedPlugin } from "@musistudio/llms";

// Start service if this file is run directly
if (require.main === module) {
  run().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}
