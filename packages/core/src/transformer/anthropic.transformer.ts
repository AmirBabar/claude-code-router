import { ChatCompletion } from "openai/resources";
import {
  LLMProvider,
  UnifiedChatRequest,
  UnifiedMessage,
  UnifiedTool,
} from "@/types/llm";
import {
  Transformer,
  TransformerContext,
  TransformerOptions,
} from "@/types/transformer";
import { v4 as uuidv4 } from "uuid";
import { getThinkLevel } from "@/utils/thinking";
import { createApiError } from "@/api/middleware";
import { formatBase64 } from "@/utils/image";

export class AnthropicTransformer implements Transformer {
  name = "Anthropic";
  endPoint = "/v1/messages";
  private useBearer: boolean;
  logger?: any;

  constructor(private readonly options?: TransformerOptions) {
    this.useBearer = this.options?.UseBearer ?? false;
  }

  private getCachedTokens(usage: any): number {
    // Try prompt_tokens_details?.cached_tokens first (OpenAI standard),
    // then prompt_cache_hit_tokens (DeepSeek-specific)
    // Using `any` type to handle both standard and non-standard cache fields
    return usage?.prompt_tokens_details?.cached_tokens ||
           usage?.prompt_cache_hit_tokens ||
           0;
  }

  async auth(request: any, provider: LLMProvider): Promise<any> {
    const headers: Record<string, string | undefined> = {};

    if (this.useBearer) {
      headers["authorization"] = `Bearer ${provider.apiKey}`;
      headers["x-api-key"] = undefined;
    } else {
      headers["x-api-key"] = provider.apiKey;
      headers["authorization"] = undefined;
    }

    return {
      body: request,
      config: {
        headers,
      },
    };
  }

  async transformRequestOut(
    request: Record<string, any>
  ): Promise<UnifiedChatRequest> {
    const messages: UnifiedMessage[] = [];

    if (request.system) {
      if (typeof request.system === "string") {
        messages.push({
          role: "system",
          content: request.system,
        });
      } else if (Array.isArray(request.system) && request.system.length) {
        const textParts = request.system
          .filter((item: any) => item.type === "text" && item.text)
          .map((item: any) => ({
            type: "text" as const,
            text: item.text,
            cache_control: item.cache_control,
          }));
        messages.push({
          role: "system",
          content: textParts,
        });
      }
    }

    const requestMessages = JSON.parse(JSON.stringify(request.messages || []));

    requestMessages?.forEach((msg: any) => {
      if (msg.role === "user" || msg.role === "assistant") {
        if (typeof msg.content === "string") {
          messages.push({
            role: msg.role,
            content: msg.content,
          });
          return;
        }

        if (Array.isArray(msg.content)) {
          if (msg.role === "user") {
            const toolParts = msg.content.filter(
              (c: any) => c.type === "tool_result" && c.tool_use_id
            );
            if (toolParts.length) {
              toolParts.forEach((tool: any) => {
                const toolMessage: UnifiedMessage = {
                  role: "tool",
                  content:
                    typeof tool.content === "string"
                      ? tool.content
                      : JSON.stringify(tool.content),
                  tool_call_id: tool.tool_use_id,
                  cache_control: tool.cache_control,
                };
                messages.push(toolMessage);
              });
            }

            const textAndMediaParts = msg.content.filter(
              (c: any) =>
                (c.type === "text" && c.text) ||
                (c.type === "image" && c.source)
            );
            if (textAndMediaParts.length) {
              messages.push({
                role: "user",
                content: textAndMediaParts.map((part: any) => {
                  if (part?.type === "image") {
                    return {
                      type: "image_url",
                      image_url: {
                        url:
                          part.source?.type === "base64"
                            ? formatBase64(
                                part.source.data,
                                part.source.media_type
                              )
                            : part.source.url,
                      },
                      media_type: part.source.media_type,
                    };
                  }
                  return part;
                }),
              });
            }
          } else if (msg.role === "assistant") {
            const assistantMessage: UnifiedMessage = {
              role: "assistant",
              content: "",
            };
            const textParts = msg.content.filter(
              (c: any) => c.type === "text" && c.text
            );
            if (textParts.length) {
              assistantMessage.content = textParts
                .map((text: any) => text.text)
                .join("\n");
            }

            const toolCallParts = msg.content.filter(
              (c: any) => c.type === "tool_use" && c.id
            );
            if (toolCallParts.length) {
              assistantMessage.tool_calls = toolCallParts.map((tool: any) => {
                return {
                  id: tool.id,
                  type: "function" as const,
                  function: {
                    name: tool.name,
                    arguments: JSON.stringify(tool.input || {}),
                  },
                };
              });
            }

            const thinkingPart = msg.content.find(
              (c: any) => c.type === "thinking" && c.signature
            );
            if (thinkingPart) {
              assistantMessage.thinking = {
                content: thinkingPart.thinking,
                signature: thinkingPart.signature,
              };
            }

            messages.push(assistantMessage);
          }
          return;
        }
      }
    });

    const result: UnifiedChatRequest = {
      messages,
      model: request.model,
      max_tokens: request.max_tokens,
      temperature: request.temperature,
      stream: request.stream,
      tools: request.tools?.length
        ? this.convertAnthropicToolsToUnified(request.tools)
        : undefined,
      tool_choice: request.tool_choice,
    };
    if (request.thinking) {
      result.reasoning = {
        effort: getThinkLevel(request.thinking.budget_tokens),
        enabled: request.thinking.type === "enabled",
      };
    }
    if (request.tool_choice) {
      if (request.tool_choice.type === "tool") {
        result.tool_choice = {
          type: "function",
          function: { name: request.tool_choice.name },
        };
      } else {
        result.tool_choice = request.tool_choice.type;
      }
    }
    return result;
  }

  async transformResponseIn(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    const isStream = response.headers
      .get("Content-Type")
      ?.includes("text/event-stream");
    if (isStream) {
      if (!response.body) {
        throw new Error("Stream response body is null");
      }
      const convertedStream = await this.convertOpenAIStreamToAnthropic(
        response.body,
        context!
      );
      return new Response(convertedStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      const data = (await response.json()) as any;
      const anthropicResponse = this.convertOpenAIResponseToAnthropic(
        data,
        context!
      );
      return new Response(JSON.stringify(anthropicResponse), {
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  private convertAnthropicToolsToUnified(tools: any[]): UnifiedTool[] {
    return tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description || "",
        parameters: tool.input_schema,
      },
    }));
  }

  private async convertOpenAIStreamToAnthropic(
    openaiStream: ReadableStream,
    context: TransformerContext
  ): Promise<ReadableStream> {
    const readable = new ReadableStream({
      start: async (controller) => {
        const encoder = new TextEncoder();
        const messageId = `msg_${Date.now()}`;
        let stopReasonMessageDelta: null | Record<string, any> = null;
        let model = "unknown";
        let hasStarted = false;
        let hasTextContentStarted = false;
        let hasFinished = false;
        const toolCalls = new Map<number, any>();
        const toolCallIndexToContentBlockIndex = new Map<number, number>();
        let contentIndex = 0;
        let isClosed = false;
        let isThinkingStarted = false;
        let currentContentBlockIndex = -1;

        const assignContentBlockIndex = (): number => {
          const currentIndex = contentIndex;
          contentIndex++;
          return currentIndex;
        };

        const safeEnqueue = (data: Uint8Array) => {
          if (!isClosed) {
            try {
              controller.enqueue(data);
            } catch (error) {
              isClosed = true;
            }
          }
        };

        const safeClose = () => {
          if (!isClosed) {
            try {
              if (currentContentBlockIndex >= 0) {
                const contentBlockStop = {
                  type: "content_block_stop",
                  index: currentContentBlockIndex,
                };
                safeEnqueue(
                  encoder.encode(
                    `event: content_block_stop\ndata: ${JSON.stringify(
                      contentBlockStop
                    )}\n\n`
                  )
                );
              }

              if (stopReasonMessageDelta) {
                safeEnqueue(
                  encoder.encode(
                    `event: message_delta\ndata: ${JSON.stringify(
                      stopReasonMessageDelta
                    )}\n\n`
                  )
                );
              }
              const messageStop = { type: "message_stop" };
              safeEnqueue(
                encoder.encode(
                  `event: message_stop\ndata: ${JSON.stringify(messageStop)}\n\n`
                )
              );
              controller.close();
              isClosed = true;
            } catch (error) {
              isClosed = true;
            }
          }
        };

        let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

        try {
          reader = openaiStream.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            if (isClosed) break;
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (isClosed || hasFinished) break;
              if (!line.startsWith("data:")) continue;
              const data = line.slice(5).trim();
              if (data === "[DONE]") continue;

              try {
                const chunk = JSON.parse(data);
                if (chunk.error) {
                  const errorMessage = {
                    type: "error",
                    message: {
                      type: "api_error",
                      message: JSON.stringify(chunk.error),
                    },
                  };
                  safeEnqueue(encoder.encode(`event: error\ndata: ${JSON.stringify(errorMessage)}\n\n`));
                  continue;
                }

                model = chunk.model || model;

                if (!hasStarted && !isClosed) {
                  hasStarted = true;
                  const messageStart = {
                    type: "message_start",
                    message: {
                      id: messageId,
                      type: "message",
                      role: "assistant",
                      content: [],
                      model: model,
                      stop_reason: null,
                      stop_sequence: null,
                      usage: { input_tokens: 0, output_tokens: 0 },
                    },
                  };
                  safeEnqueue(encoder.encode(`event: message_start\ndata: ${JSON.stringify(messageStart)}\n\n`));
                }

                const choice = chunk.choices?.[0];
                if (chunk.usage) {
                  const cached = this.getCachedTokens(chunk.usage);
                  stopReasonMessageDelta = {
                    type: "message_delta",
                    delta: { stop_reason: "end_turn", stop_sequence: null },
                    usage: {
                      input_tokens: (chunk.usage?.prompt_tokens || 0) - cached,
                      output_tokens: chunk.usage?.completion_tokens || 0,
                      cache_read_input_tokens: cached,
                    },
                  };
                }

                if (!choice) continue;

                // --- THINKING LOGIC ---
                if (choice?.delta?.thinking && !isClosed) {
                  if (!isThinkingStarted) {
                    const thinkingBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: thinkingBlockIndex,
                      content_block: { type: "thinking", thinking: "" },
                    };
                    safeEnqueue(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify(contentBlockStart)}\n\n`));
                    currentContentBlockIndex = thinkingBlockIndex;
                    isThinkingStarted = true;
                  }
                  if (choice.delta.thinking.signature) {
                    safeEnqueue(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                      type: "content_block_delta",
                      index: currentContentBlockIndex,
                      delta: { type: "signature_delta", signature: choice.delta.thinking.signature }
                    })}\n\n`));
                    safeEnqueue(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: currentContentBlockIndex })}\n\n`));
                    currentContentBlockIndex = -1;
                  } else if (choice.delta.thinking.content) {
                    safeEnqueue(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                      type: "content_block_delta",
                      index: currentContentBlockIndex,
                      delta: { type: "thinking_delta", thinking: choice.delta.thinking.content }
                    })}\n\n`));
                  }
                }

                // --- TEXT CONTENT LOGIC ---
                if (choice?.delta?.content && !isClosed) {
                  if (currentContentBlockIndex >= 0 && !hasTextContentStarted) {
                    safeEnqueue(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: currentContentBlockIndex })}\n\n`));
                    currentContentBlockIndex = -1;
                  }
                  if (!hasTextContentStarted) {
                    hasTextContentStarted = true;
                    const textBlockIndex = assignContentBlockIndex();
                    safeEnqueue(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
                      type: "content_block_start",
                      index: textBlockIndex,
                      content_block: { type: "text", text: "" }
                    })}\n\n`));
                    currentContentBlockIndex = textBlockIndex;
                  }
                  safeEnqueue(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                    type: "content_block_delta",
                    index: currentContentBlockIndex,
                    delta: { type: "text_delta", text: choice.delta.content }
                  })}\n\n`));
                }

                // --- CITATIONS / ANNOTATIONS LOGIC (PERPLEXITY & DEEPSEEK) ---
                const citations = choice?.delta?.annotations || choice?.delta?.citations;
                if (citations?.length && !isClosed) {
                  if (currentContentBlockIndex >= 0 && hasTextContentStarted) {
                    safeEnqueue(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: currentContentBlockIndex })}\n\n`));
                    currentContentBlockIndex = -1;
                    hasTextContentStarted = false;
                  }
                  citations.forEach((item: any, idx: number) => {
                    const isPerplexity = typeof item === 'string';
                    const url = isPerplexity ? item : item.url_citation?.url;
                    const title = isPerplexity ? `Source ${idx + 1}` : item.url_citation?.title;
                    const blockIndex = assignContentBlockIndex();
                    safeEnqueue(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
                      type: "content_block_start",
                      index: blockIndex,
                      content_block: {
                        type: "web_search_tool_result",
                        tool_use_id: `srvtoolu_${uuidv4()}`,
                        content: [{ type: "web_search_result", title, url }]
                      }
                    })}\n\n`));
                    safeEnqueue(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: blockIndex })}\n\n`));
                  });
                }

                // --- TOOL CALLS LOGIC ---
                if (choice?.delta?.tool_calls && !isClosed) {
                  for (const toolCall of choice.delta.tool_calls) {
                    const toolCallIndex = toolCall.index ?? 0;
                    if (!toolCallIndexToContentBlockIndex.has(toolCallIndex)) {
                      if (currentContentBlockIndex >= 0) {
                        safeEnqueue(encoder.encode(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: currentContentBlockIndex })}\n\n`));
                      }
                      const newBlockIndex = assignContentBlockIndex();
                      toolCallIndexToContentBlockIndex.set(toolCallIndex, newBlockIndex);
                      safeEnqueue(encoder.encode(`event: content_block_start\ndata: ${JSON.stringify({
                        type: "content_block_start",
                        index: newBlockIndex,
                        content_block: { type: "tool_use", id: toolCall.id || `call_${Date.now()}`, name: toolCall.function?.name || "tool", input: {} }
                      })}\n\n`));
                      currentContentBlockIndex = newBlockIndex;
                    }
                    if (toolCall.function?.arguments) {
                      safeEnqueue(encoder.encode(`event: content_block_delta\ndata: ${JSON.stringify({
                        type: "content_block_delta",
                        index: toolCallIndexToContentBlockIndex.get(toolCallIndex),
                        delta: { type: "input_json_delta", partial_json: toolCall.function.arguments }
                      })}\n\n`));
                    }
                  }
                }

                if (choice?.finish_reason && !isClosed) {
                  hasFinished = true;
                  break;
                }
              } catch (parseError) {}
            }
          }
          safeClose();
        } catch (error) {
          if (!isClosed) controller.error(error);
        } finally {
          if (reader) reader.releaseLock();
        }
      },
      cancel: (reason) => {}
    });
    return readable;
  }

  private convertOpenAIResponseToAnthropic(
    openaiResponse: ChatCompletion,
    context: TransformerContext
  ): any {
    try {
      const choice = openaiResponse.choices[0];
      if (!choice) throw new Error("No choices found");
      const content: any[] = [];
      const annotations = choice.message.annotations || (choice.message as any).citations;

      if (annotations?.length) {
        const id = `srvtoolu_${uuidv4()}`;
        content.push({
          type: "web_search_tool_result",
          tool_use_id: id,
          content: annotations.map((item: any) => {
            const isPerp = typeof item === 'string';
            return {
              type: "web_search_result",
              url: isPerp ? item : item.url_citation.url,
              title: isPerp ? "Source" : item.url_citation.title,
            };
          }),
        });
      }
      if (choice.message.content) content.push({ type: "text", text: choice.message.content });
      if ((choice.message as any)?.thinking?.content) {
        content.push({
          type: "thinking",
          thinking: (choice.message as any).thinking.content,
          signature: (choice.message as any).thinking.signature,
        });
      }
      
      const cached = this.getCachedTokens(openaiResponse.usage);
      
      return {
        id: openaiResponse.id,
        type: "message",
        role: "assistant",
        model: openaiResponse.model,
        content: content,
        stop_reason: choice.finish_reason === "stop" ? "end_turn" : "tool_use",
        stop_sequence: null,
        usage: {
          input_tokens: (openaiResponse.usage?.prompt_tokens || 0) - cached,
          output_tokens: openaiResponse.usage?.completion_tokens || 0,
          cache_read_input_tokens: cached,
        },
      };
    } catch {
      throw createApiError("Provider error", 500, "provider_error");
    }
  }
}