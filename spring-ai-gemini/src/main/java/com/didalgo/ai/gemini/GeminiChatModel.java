package com.didalgo.ai.gemini;

import com.didalgo.ai.gemini.api.GeminiApi;
import com.didalgo.ai.gemini.api.GeminiApi.Content;
import com.didalgo.ai.gemini.api.GeminiApi.FunctionCall;
import com.didalgo.ai.gemini.api.GeminiApi.FunctionCallingConfig;
import com.didalgo.ai.gemini.api.GeminiApi.FunctionDeclaration;
import com.didalgo.ai.gemini.api.GeminiApi.GenerateContentRequest;
import com.didalgo.ai.gemini.api.GeminiApi.GenerateContentResponse;
import com.didalgo.ai.gemini.api.GeminiApi.GenerationConfig;
import com.didalgo.ai.gemini.api.GeminiApi.Part;
import com.didalgo.ai.gemini.api.GeminiApi.Schema;
import com.didalgo.ai.gemini.api.GeminiApi.Tool;
import com.didalgo.ai.gemini.api.GeminiApi.ToolConfig;
import com.didalgo.ai.gemini.metadata.GeminiChatResponseMetadata;
import com.didalgo.ai.gemini.metadata.GeminiUsage;
import com.fasterxml.jackson.databind.JsonNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Media;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.StreamingChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.function.AbstractFunctionCallSupport;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.NonNull;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import reactor.core.publisher.Flux;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class GeminiChatModel
        extends AbstractFunctionCallSupport<Content, GeminiChatModel.GeminiRequest, ResponseEntity<GenerateContentResponse>>
        implements ChatModel, StreamingChatModel {

    private static final Logger logger = LoggerFactory.getLogger(GeminiChatModel.class);

    /**
     * The default options used for the chat completion requests.
     */
    private GeminiChatOptions defaultOptions;

    /**
     * The retry template used to retry the Gemini API calls.
     */
    private final RetryTemplate retryTemplate;

    /**
     * Low-level access to the Gemini API.
     */
    private final GeminiApi geminiApi;

    /**
     * The default generating options used for the chat completion requests.
     */
    private final GenerationConfig generationConfig;

    /**
     * Creates an instance of the GeminiChatClient.
     *
     * @param geminiApi the {@code GeminiApi} instance to be used for interacting with the Google
     * Gemini API
     * @throws IllegalArgumentException if geminiApi is null
     */
    public GeminiChatModel(GeminiApi geminiApi) {
        this(geminiApi,
                GeminiChatOptions.builder().withModel(GeminiApi.DEFAULT_CHAT_MODEL).withTemperature(0.7f).build());
    }

    /**
     * Creates an instance of the GeminiChatClient.
     *
     * @param geminiApi the {@code GeminiApi} instance to be used for interacting with the Google
     * Gemini API
     * @param options the {@code GeminiChatOptions} to configure the chat client
     */
    public GeminiChatModel(GeminiApi geminiApi, GeminiChatOptions options) {
        this(geminiApi, options, null, RetryUtils.DEFAULT_RETRY_TEMPLATE);
    }

    /**
     * Initializes a new instance of the GeminiChatClient.
     *
     * @param geminiApi the {@code GeminiApi} instance to be used for interacting with the Google
     * Gemini API
     * @param options the {@code GeminiChatOptions} to configure the chat client
     * @param functionCallbackContext The function callback context.
     * @param retryTemplate The retry template.
     */
    public GeminiChatModel(GeminiApi geminiApi, GeminiChatOptions options,
                           FunctionCallbackContext functionCallbackContext, RetryTemplate retryTemplate) {
        super(functionCallbackContext);
        Assert.notNull(geminiApi, "GeminiApi must not be null");
        Assert.notNull(options, "Options must not be null");
        Assert.notNull(retryTemplate, "RetryTemplate must not be null");
        this.geminiApi = geminiApi;
        this.defaultOptions = options;
        this.retryTemplate = retryTemplate;
        this.generationConfig = toGenerationConfig(options);
    }

    public record GeminiRequest(String model, GenerateContentRequest request) {}

    @Override
    public ChatResponse call(Prompt prompt) {
        GeminiRequest geminiRequest = createGeminiRequest(prompt);

        return this.retryTemplate.execute(ctx -> {
            ResponseEntity<GenerateContentResponse> response = this.callWithFunctionSupport(geminiRequest);
            List<Generation> generations = Optional.ofNullable(response.getBody().candidates()).orElse(List.of())
                    .stream()
                    .map(candidate -> candidate.content().parts())
                    .flatMap(List::stream)
                    .map(Part::text)
                    .map(Generation::new)
                    .toList();

            return new ChatResponse(generations, toChatResponseMetadata(response.getBody()));
        });
    }

    @Override
    public Flux<ChatResponse> stream(Prompt prompt) {
        GeminiRequest request = createGeminiRequest(prompt);

        return this.retryTemplate.execute(ctx -> {
            Flux<GenerateContentResponse> responseStream = geminiApi.streamGenerateContent(request.model(), request.request());

            return responseStream
                    .switchMap(r -> handleFunctionCallOrReturnStream(request, Flux.just(ResponseEntity.of(Optional.of(r)))))
                    .map(ResponseEntity::getBody)
                    .map(response -> {
                        List<Generation> generations = response.candidates()
                                .stream()
                                .map(candidate -> candidate.content().parts())
                                .flatMap(List::stream)
                                .map(Part::text)
                                .map(Generation::new)
                                .toList();

                        return new ChatResponse(generations, toChatResponseMetadata(response));
                    });
        });
    }

    private GeminiChatResponseMetadata toChatResponseMetadata(GenerateContentResponse response) {
        return new GeminiChatResponseMetadata(new GeminiUsage(response.usageMetadata()));
    }

    private GeminiRequest createGeminiRequest(Prompt prompt) {
        Set<String> functionsForThisRequest = new HashSet<>();
        GenerationConfig generationConfig = this.generationConfig;

        String modelName = this.defaultOptions.getModel();
        //var generativeModelBuilder = new GenerativeModel.Builder().setModelName(this.defaultOptions.getModel())
        //        .setVertexAi(this.vertexAI);

        GeminiChatOptions updatedRuntimeOptions = null;
        if (prompt.getOptions() != null) {
            if (prompt.getOptions() instanceof ChatOptions runtimeOptions) {
                updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(runtimeOptions, ChatOptions.class, GeminiChatOptions.class);

                functionsForThisRequest
                        .addAll(handleFunctionCallbackConfigurations(updatedRuntimeOptions, IS_RUNTIME_CALL));
            } else {
                throw new IllegalArgumentException("Prompt options are not of type ChatOptions: "
                        + prompt.getOptions().getClass().getSimpleName());
            }
        }

        if (this.defaultOptions != null) {
            functionsForThisRequest.addAll(handleFunctionCallbackConfigurations(this.defaultOptions, !IS_RUNTIME_CALL));

            if (updatedRuntimeOptions == null) {
                updatedRuntimeOptions = GeminiChatOptions.builder().build();
            }
            updatedRuntimeOptions = ModelOptionsUtils.merge(updatedRuntimeOptions, this.defaultOptions, GeminiChatOptions.class);
        }

        if (updatedRuntimeOptions != null) {
            if (StringUtils.hasText(updatedRuntimeOptions.getModel())
                    && !updatedRuntimeOptions.getModel().equals(this.defaultOptions.getModel())) {

                // Override model name
                modelName = updatedRuntimeOptions.getModel();
            }
            generationConfig = toGenerationConfig(updatedRuntimeOptions);
        }

        // Add the enabled functions definitions to the request's tools parameter.
        List<Tool> tools = null;
        if (!CollectionUtils.isEmpty(functionsForThisRequest)) {
            tools = this.getFunctionTools(functionsForThisRequest);
        }

        // Add optional Tool Config
        ToolConfig.Builder toolConfigBuilder = null;
        if (updatedRuntimeOptions != null) {
            if (updatedRuntimeOptions.getFunctionCallingMode() != null) {
                toolConfigBuilder = ToolConfig.builder();
                toolConfigBuilder.functionCallingConfig(new FunctionCallingConfig(updatedRuntimeOptions.getFunctionCallingMode()));
            }
        }

        GenerateContentRequest contentRequest = new GenerateContentRequest(
                toGeminiContent(prompt),
                tools,
                (toolConfigBuilder == null) ? null : toolConfigBuilder.build(),
                null,
                toGeminiSystemInstruction(prompt),
                generationConfig
        );
        return new GeminiRequest(modelName, contentRequest);
    }

    private List<Tool> getFunctionTools(Set<String> functionNames) {
        List<FunctionDeclaration> functionDeclarations = this.resolveFunctionCallbacks(functionNames)
                .stream()
                .map(functionCallback -> new FunctionDeclaration(
                        functionCallback.getName(),
                        functionCallback.getDescription(),
                        jsonToSchema(functionCallback.getInputTypeSchema())))
                .toList();

        return List.of(new Tool(functionDeclarations));
    }

    private static String structToJson(JsonNode struct) {
        try {
            return ModelOptionsUtils.toJsonString(struct);
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static JsonNode jsonToStruct(String json) {
        try {
            return ModelOptionsUtils.OBJECT_MAPPER.readTree(json);
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static Schema jsonToSchema(String json) {
        try {
            return ModelOptionsUtils.jsonToObject(json, Schema.class);
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private GenerationConfig toGenerationConfig(GeminiChatOptions options) {
        GenerationConfig.Builder builder = GenerationConfig.builder();
        if (options.getTemperature() != null) {
            builder.temperature(options.getTemperature().doubleValue());
        }
        if (options.getMaxOutputTokens() != null) {
            builder.maxOutputTokens(options.getMaxOutputTokens());
        }
        if (options.getTopK() != null) {
            builder.topK(options.getTopK());
        }
        if (options.getTopP() != null) {
            builder.topP(options.getTopP().doubleValue());
        }
        if (options.getCandidateCount() != null) {
            builder.candidateCount(options.getCandidateCount());
        }
        if (options.getStopSequences() != null) {
            builder.stopSequences(options.getStopSequences());
        }

        return builder.build();
    }

    private Content toGeminiSystemInstruction(Prompt prompt) {
        String systemContext = prompt.getInstructions()
                .stream()
                .filter(m -> m.getMessageType() == MessageType.SYSTEM)
                .map(m -> m.getContent())
                .collect(Collectors.joining(System.lineSeparator()));

        return StringUtils.hasText(systemContext) ? new Content(null, List.of(Part.fromText(systemContext))) : null;
    }

    private List<Content> toGeminiContent(Prompt prompt) {
        List<Content> contents = prompt.getInstructions()
                .stream()
                .filter(m -> m.getMessageType() == MessageType.USER || m.getMessageType() == MessageType.ASSISTANT)
                .map(message -> new Content(toGeminiMessageType(message.getMessageType()).id(), messageToGeminiParts(message)))
                .toList();

        return contents;
    }

    private static Content.Role toGeminiMessageType(@NonNull MessageType type) {
        Assert.notNull(type, "Message type must not be null");

        return switch (type) {
            case USER -> Content.Role.USER;
            case ASSISTANT -> Content.Role.MODEL;
            default -> throw new IllegalArgumentException("Unsupported message type: " + type);
        };
    }

    private static List<Part> messageToGeminiParts(Message message) {
        if (message instanceof UserMessage userMessage) {

            String messageTextContent = (userMessage.getContent() == null) ? "null" : userMessage.getContent();
            Part textPart = Part.builder().text(messageTextContent).build();
            List<Part> parts = new ArrayList<>(List.of(textPart));
            List<Part> mediaParts = userMessage.getMedia()
                    .stream()
                    .map(GeminiChatModel::mediaToGeminiPart)
                    .toList();
            if (!CollectionUtils.isEmpty(mediaParts)) {
                parts.addAll(mediaParts);
            }

            return parts;
        }
        else if (message instanceof AssistantMessage assistantMessage) {
            return List.of(Part.builder().text(assistantMessage.getContent()).build());
        }
        else {
            throw new IllegalArgumentException("Gemini doesn't support message type: " + message.getClass());
        }
    }

    private static Part mediaToGeminiPart(Media media) {
        if (media.getData() instanceof byte[] bytes) {
            return Part.builder()
                    .inlineData(GeminiApi.Blob.from(media.getMimeType().toString(), bytes))
                    .build();
        }
        else {
            throw new IllegalArgumentException("The second element of the input List can only be one of the following format: byte[]");
        }
    }

    @Override
    protected GeminiRequest doCreateToolResponseRequest(GeminiRequest previousRequest, Content responseMessage, List<Content> conversationHistory) {
        FunctionCall functionCall = responseMessage.parts().get(0).functionCall();

        String functionName = functionCall.name();
        String functionArguments = structToJson(functionCall.args());
        if (!functionCallbackRegister.containsKey(functionName)) {
            throw new IllegalStateException("No function callback found for function name: " + functionName);
        }

        String functionResponse = functionCallbackRegister.get(functionName).call(functionArguments);
        Content contentFnResp = Content.fromPart(
                Part.fromFunctionResponse(functionCall.name(), jsonToStruct(functionResponse))
        );
        conversationHistory.add(contentFnResp);

        return new GeminiRequest(previousRequest.model(), previousRequest.request().withContents(conversationHistory));
    }

    @Override
    protected List<Content> doGetUserMessages(GeminiRequest request) {
        return request.request().contents();
    }

    @Override
    protected Content doGetToolResponseMessage(ResponseEntity<GenerateContentResponse> response) {
        return response.getBody().candidates().get(0).content();
    }

    @Override
    protected ResponseEntity<GenerateContentResponse> doChatCompletion(GeminiRequest request) {
        try {
            return geminiApi.generateContent(request.model(), request.request());
        }
        catch (Exception e) {
            throw new RuntimeException("Failed to generate content", e);
        }
    }

    @Override
    protected Flux<ResponseEntity<GenerateContentResponse>> doChatCompletionStream(GeminiRequest request) {
        return geminiApi.streamGenerateContent(request.model(), request.request())
                //.map(this::chunkToChatCompletion)
                .map(Optional::ofNullable)
                .map(ResponseEntity::of);
    }

    @Override
    protected boolean isToolFunctionCall(ResponseEntity<GenerateContentResponse> response) {
        var body = response.getBody();
        if (body == null || CollectionUtils.isEmpty(body.candidates())
                || body.candidates().get(0).content() == null
                || CollectionUtils.isEmpty(body.candidates().get(0).content().parts())) {
            return false;
        }
        return body.candidates().get(0).content().parts().get(0).hasFunctionCall();
    }

    @Override
    public ChatOptions getDefaultOptions() {
        return GeminiChatOptions.fromOptions(this.defaultOptions);
    }
}
