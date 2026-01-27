import { UnifiedChatRequest } from "@/types/llm";
import {
  Transformer,
  TransformerContext,
  TransformerOptions,
} from "@/types/transformer";

export class PerplexityTransformer implements Transformer {
  name = "Perplexity";
  endPoint = "/chat/completions";

  constructor(private readonly options?: TransformerOptions) {}

  async auth(request: any, provider: any): Promise<any> {
    return {
      body: request,
      config: {
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
          "Content-Type": "application/json",
        },
      },
    };
  }

  async transformRequestOut(
    request: any, 
    context: TransformerContext
  ): Promise<UnifiedChatRequest> {
    // Explicitly cast or construct to satisfy UnifiedChatRequest requirements
    const transformed: UnifiedChatRequest = {
      model: request.model,
      messages: request.messages,
      max_tokens: request.max_tokens,
      temperature: request.temperature,
      stream: request.stream,
      // Pass through tools if they exist
      tools: request.tools,
      tool_choice: request.tool_choice
    };

    // Force Perplexity-specific citation requirement
    (transformed as any).return_citations = true; 

    return transformed;
  }

  async transformResponseIn(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    // Pass-through: downstream Anthropic transformer handles parsing
    return response;
  }
}