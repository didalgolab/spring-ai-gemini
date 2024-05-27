package com.didalgo.ai.gemini.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.util.Assert;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class GeminiApi {

    public static final String DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com";

    public static final String DEFAULT_CHAT_MODEL = GeminiApi.ChatModel.GEMINI_1_5_FLASH_LATEST.getId();

    private final RestClient restClient;

    private final WebClient webClient;


    public GeminiApi(String apiKey) {
        this(DEFAULT_BASE_URL, apiKey);
    }

    public GeminiApi(String baseUrl, String apiKey) {
        this(baseUrl, apiKey, RestClient.builder());
    }

    public GeminiApi(String baseUrl, String apiKey, RestClient.Builder restClientBuilder) {
        this(baseUrl, apiKey, restClientBuilder, RetryUtils.DEFAULT_RESPONSE_ERROR_HANDLER);
    }

    public GeminiApi(String baseUrl, String apiKey, RestClient.Builder restClientBuilder, ResponseErrorHandler responseErrorHandler) {
        this.restClient = restClientBuilder
                .baseUrl(baseUrl)
                .defaultHeaders(ApiUtils.getJsonContentHeaders(apiKey))
                .defaultStatusHandler(responseErrorHandler)
                .build();
        this.webClient = WebClient.builder()
                .baseUrl(baseUrl)
                .defaultHeaders(ApiUtils.getJsonContentHeaders(apiKey))
                .build();
    }

    public enum ChatModel {
        GEMINI_1_5_FLASH_LATEST("gemini-1.5-flash-latest"),
        GEMINI_1_5_PRO_LATEST("gemini-1.5-pro-latest");

        final String id;

        ChatModel(String id) {
            this.id = id;
        }

        public String getId() {
            return id;
        }
    }

    /**
     * Represents a request to generate content using the Gemini API.
     *
     * @param contents Required. The content of the current conversation with the model.
     *                 For single-turn queries, this is a single instance. For multi-turn
     *                 queries, this is a repeated field that contains conversation history
     *                 and the latest request.
     * @param tools Optional. A list of {@code Tool} instances the model may use to generate
     *              the next response. A {@code Tool} is a piece of code that enables the
     *              system to interact with external systems to perform an action, or set of
     *              actions, outside of the knowledge and scope of the model. The only
     *              supported tool is currently {@code Function}.
     * @param toolConfig Optional. Tool configuration for any {@code Tool} specified in the
     *                   request.
     * @param safetySettings Optional. A list of unique {@code SafetySetting} instances for
     *                       blocking unsafe content. This will be enforced on the
     *                       {@code GenerateContentRequest.contents} and
     *                       {@code GenerateContentResponse.candidates}. There should not be
     *                       more than one setting for each {@code SafetyCategory} type. The
     *                       API will block any contents and responses that fail to meet the
     *                       thresholds set by these settings. This list overrides the
     *                       default settings for each {@code SafetyCategory} specified in
     *                       the {@code safetySettings}. If there is no {@code SafetySetting}
     *                       for a given {@code SafetyCategory} provided in the list, the API
     *                       will use the default safety setting for that category. Harm
     *                       categories {@code HARM_CATEGORY_HATE_SPEECH},
     *                       {@code HARM_CATEGORY_SEXUALLY_EXPLICIT},
     *                       {@code HARM_CATEGORY_DANGEROUS_CONTENT},
     *                       {@code HARM_CATEGORY_HARASSMENT} are supported.
     * @param systemInstruction Optional. Developer set system instruction. Currently, text
     *                          only.
     * @param generationConfig Optional. Configuration options for model generation and
     *                         outputs.
     */
    public record GenerateContentRequest(
            @JsonProperty("contents") List<Content> contents,
            @JsonProperty("tools") List<Tool> tools,
            @JsonProperty("toolConfig") ToolConfig toolConfig,
            @JsonProperty("safetySettings") List<SafetySetting> safetySettings,
            @JsonProperty("systemInstruction") Content systemInstruction,
            @JsonProperty("generationConfig") GenerationConfig generationConfig
    ) {
        public GenerateContentRequest withContents(List<Content> contents) {
            Assert.notNull(contents, "contents must not be null");
            return new GenerateContentRequest(contents, tools, toolConfig, safetySettings, systemInstruction, generationConfig);
        }
    }

    /**
     * Represents the content of a message in a conversation.
     *
     * @param parts Ordered {@code Part} instances that constitute a single message.
     *              Parts may have different MIME types.
     * @param role Optional. The producer of the content. Must be either {@code "user"}
     *             or {@code "model"}. Useful to set for multi-turn conversations,
     *             otherwise can be left blank or unset.
     */
    public record Content(
            @JsonProperty("role") String role,
            @JsonProperty("parts") List<Part> parts
    ) {

        public static Content fromPart(Part part) {
            return new Content(null, List.of(part));
        }

        public static Content fromParts(List<Part> parts) {
            return new Content(null, parts);
        }

        public enum Role {
            USER("user"),
            MODEL("model");

            final String id;

            Role(String id) {
                this.id = id;
            }

            public String id() {
                return id;
            }
        }
    }

    /**
     * Represents a part of a message in a conversation.
     *
     * @param text Inline text.
     * @param inlineData Inline media bytes.
     * @param functionCall A predicted {@code FunctionCall} returned from the model that contains a string
     *                     representing the {@code FunctionDeclaration.name} with the arguments and their values.
     * @param functionResponse The result output of a {@code FunctionCall} that contains a string
     *                         representing the {@code FunctionDeclaration.name} and a structured JSON object
     *                         containing any output from the function used as context to the model.
     * @param fileData URI based data.
     */
    public record Part(
            @JsonProperty("text") String text,
            @JsonProperty("inlineData") Blob inlineData,
            @JsonProperty("functionCall") FunctionCall functionCall,
            @JsonProperty("functionResponse") FunctionResponse functionResponse,
            @JsonProperty("fileData") FileData fileData
    ) {

        public boolean hasFunctionCall() {
            return functionCall != null;
        }

        /**
         * Constructs a new {@code Part} for the given function response.
         *
         * @param functionName the function name
         * @param functionResponse the function response in JSON format
         * @return a new {@code Part} instance.
         */
        public static Part fromFunctionResponse(String functionName, JsonNode functionResponse) {
            return builder()
                    .functionResponse(new FunctionResponse(functionName, functionResponse))
                    .build();
        }

        public static Part fromText(String text) {
            return builder().text(text).build();
        }

        /**
         * Creates a new builder for {@code Part}.
         *
         * @return a new {@code Builder} instance.
         */
        public static Builder builder() {
            return new Builder();
        }

        /**
         * Builder class for {@code Part}.
         */
        public static class Builder {
            private String text;
            private Blob inlineData;
            private FunctionCall functionCall;
            private FunctionResponse functionResponse;
            private FileData fileData;

            public Builder text(String text) {
                this.text = text;
                return this;
            }

            public Builder inlineData(Blob inlineData) {
                this.inlineData = inlineData;
                return this;
            }

            public Builder functionCall(FunctionCall functionCall) {
                this.functionCall = functionCall;
                return this;
            }

            public Builder functionResponse(FunctionResponse functionResponse) {
                this.functionResponse = functionResponse;
                return this;
            }

            public Builder fileData(FileData fileData) {
                this.fileData = fileData;
                return this;
            }

            public Part build() {
                return new Part(
                        text,
                        inlineData,
                        functionCall,
                        functionResponse,
                        fileData
                );
            }
        }
    }

    /**
     * Represents inline media bytes with MIME type information.
     *
     * @param mimeType The IANA standard MIME type of the source data. Examples: {@code "image/png"},
     *                 {@code "image/jpeg"}. If an unsupported MIME type is provided, an error will be
     *                 returned. For a complete list of supported types, see Supported file formats.
     * @param data Raw bytes for media formats, encoded as a base64-encoded string.
     */
    public record Blob(
            @JsonProperty("mimeType") String mimeType,
            @JsonProperty("data") String data
    ) {

        /**
         * Creates a Blob instance from the given MIME type and raw bytes.
         *
         * @param mimeType The IANA standard MIME type of the source data.
         * @param bytes Raw bytes for media formats.
         * @return A new Blob instance with the provided MIME type and base64-encoded data.
         */
        public static Blob from(String mimeType, byte[] bytes) {
            String encodedData = Base64.getEncoder().encodeToString(bytes);
            return new Blob(mimeType, encodedData);
        }
    }

    /**
     * Represents a predicted function call returned from the model.
     *
     * @param name Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain
     *             underscores and dashes, with a maximum length of 63.
     * @param args Optional. The function parameters and values in JSON object format.
     */
    public record FunctionCall(
            @JsonProperty("name") String name,
            @JsonProperty("args") JsonNode args
    ) { }

    /**
     * Represents the result output from a {@code FunctionCall} that contains a string representing the
     * {@code FunctionDeclaration.name} and a structured JSON object containing any output from the function
     * used as context to the model.
     *
     * @param name Required. The name of the function to call. Must be a-z, A-Z, 0-9, or contain
     *             underscores and dashes, with a maximum length of 63.
     * @param response Required. The function response in JSON object format.
     */
    public record FunctionResponse(
            @JsonProperty("name") String name,
            @JsonProperty("response") JsonNode response
    ) { }

    /**
     * Represents URI based data.
     *
     * @param mimeType Optional. The IANA standard MIME type of the source data.
     * @param fileUri Required. The URI of the source data.
     */
    public record FileData(
            @JsonProperty("mimeType") String mimeType,
            @JsonProperty("fileUri") String fileUri
    ) { }

    /**
     * Represents a tool that the model may use to generate a response.
     *
     * <p>A Tool is a piece of code that enables the system to interact with external systems to perform an action,
     * or set of actions, outside of the knowledge and scope of the model.</p>
     *
     * @param functionDeclarations Optional. A list of {@code FunctionDeclaration} instances available to the model
     *                             that can be used for function calling. The model or system does not execute the
     *                             function. Instead, the defined function may be returned as a {@code FunctionCall}
     *                             with arguments to the client side for execution. The model may decide to call a
     *                             subset of these functions by populating {@code FunctionCall} in the response.
     *                             The next conversation turn may contain a {@code FunctionResponse} with the
     *                             {@code content.role} "function" generation context for the next model turn.
     */
    public record Tool(
            @JsonProperty("functionDeclarations") List<FunctionDeclaration> functionDeclarations
    ) { }

    /**
     * Represents a structured representation of a function declaration as defined by the OpenAPI 3.03 specification.
     * Included in this declaration are the function name and parameters. This {@code FunctionDeclaration} is a
     * representation of a block of code that can be used as a Tool by the model and executed by the client.
     *
     * @param name Required. The name of the function. Must be a-z, A-Z, 0-9, or contain underscores and dashes,
     *             with a maximum length of 63.
     * @param description Required. A brief description of the function.
     * @param parameters Optional. Describes the parameters to this function. Reflects the OpenAPI 3.03 Parameter Object.
     *                   Key: the name of the parameter. Parameter names are case sensitive.
     *                   Value: the Schema defining the type used for the parameter.
     */
    public record FunctionDeclaration(
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("parameters") Schema parameters
    ) { }

    /**
     * Represents the Tool configuration containing parameters for specifying Tool use in the request.
     *
     * @param functionCallingConfig Optional. Function calling config.
     */
    public record ToolConfig(
            @JsonProperty("functionCallingConfig") FunctionCallingConfig functionCallingConfig
    ) {
        /**
         * Creates a new builder for {@code ToolConfig}.
         *
         * @return a new {@code Builder} instance.
         */
        public static Builder builder() {
            return new Builder();
        }

        /**
         * Builder class for {@code ToolConfig}.
         */
        public static class Builder {
            private FunctionCallingConfig functionCallingConfig;

            public Builder functionCallingConfig(FunctionCallingConfig functionCallingConfig) {
                this.functionCallingConfig = functionCallingConfig;
                return this;
            }

            public ToolConfig build() {
                return new ToolConfig(functionCallingConfig);
            }
        }
    }

    /**
     * Configuration for specifying function calling behavior.
     *
     * @param mode Optional. Specifies the mode in which function calling should execute.
     *             If unspecified, the default value will be set to {@link FunctionCallingMode#AUTO}.
     * @param allowedFunctionNames Optional. A set of function names that, when provided, limits the functions
     *                             the model will call. This should only be set when the {@code mode} is {@link FunctionCallingMode#ANY}.
     *                             Function names should match {@link FunctionDeclaration#name}. With {@code mode} set to
     *                             {@link FunctionCallingMode#ANY}, the model will predict a function call from the set of function names provided.
     */
    public record FunctionCallingConfig(
            @JsonProperty("mode") FunctionCallingMode mode,
            @JsonProperty("allowedFunctionNames") List<String> allowedFunctionNames
    ) {
        /**
         * Constructs a FunctionCallingConfig with the specified mode. Useful when function names are not restricted.
         *
         * @param mode the mode in which function calling should execute
         */
        public FunctionCallingConfig(FunctionCallingMode mode) {
            this(mode, null);
        }
    }

    /**
     * Enum representing the mode in which function calling should execute.
     */
    public enum FunctionCallingMode {
        /**
         * Unspecified function calling mode. This value should not be used.
         */
        @JsonProperty("MODE_UNSPECIFIED") MODE_UNSPECIFIED,

        /**
         * Default model behavior, model decides to predict either a function call or a natural language response.
         */
        @JsonProperty("AUTO") AUTO,

        /**
         * Model is constrained to always predicting a function call only. If {@code allowedFunctionNames} are set,
         * the predicted function call will be limited to any one of {@code allowedFunctionNames}, else the predicted
         * function call will be any one of the provided {@code functionDeclarations}.
         */
        @JsonProperty("ANY") ANY,

        /**
         * Model will not predict any function call. Model behavior is same as when not passing any function declarations.
         */
        @JsonProperty("NONE") NONE
    }

    /**
     * Represents a safety setting, affecting the safety-blocking behavior.
     *
     * <p>Passing a safety setting for a category changes the allowed probability that content is blocked.</p>
     *
     * @param category Required. The category for this setting.
     * @param threshold Required. Controls the probability threshold at which harm is blocked.
     */
    public record SafetySetting(
            @JsonProperty("category") HarmCategory category,
            @JsonProperty("threshold") HarmBlockThreshold threshold
    ) {
        /**
         * Enum representing the category for a safety setting.
         */
        public enum HarmCategory {
            /** Category is unspecified. */
            @JsonProperty HARM_CATEGORY_UNSPECIFIED,
            /** Negative or harmful comments targeting identity and/or protected attribute. */
            @JsonProperty HARM_CATEGORY_DEROGATORY,
            /** Content that is rude, disrespectful, or profane. */
            @JsonProperty HARM_CATEGORY_TOXICITY,
            /** Describes scenarios depicting violence against an individual or group, or general descriptions of gore. */
            @JsonProperty HARM_CATEGORY_VIOLENCE,
            /** Contains references to sexual acts or other lewd content. */
            @JsonProperty HARM_CATEGORY_SEXUAL,
            /** Promotes unchecked medical advice. */
            @JsonProperty HARM_CATEGORY_MEDICAL,
            /** Dangerous content that promotes, facilitates, or encourages harmful acts. */
            @JsonProperty HARM_CATEGORY_DANGEROUS,
            /** Harassment content. */
            @JsonProperty HARM_CATEGORY_HARASSMENT,
            /** Hate speech and content. */
            @JsonProperty HARM_CATEGORY_HATE_SPEECH,
            /** Sexually explicit content. */
            @JsonProperty HARM_CATEGORY_SEXUALLY_EXPLICIT,
            /** Dangerous content. */
            @JsonProperty HARM_CATEGORY_DANGEROUS_CONTENT
        }

        /**
         * Enum representing the probability threshold at which harm is blocked.
         */
        public enum HarmBlockThreshold {
            /** Threshold is unspecified. */
            @JsonProperty("HARM_BLOCK_THRESHOLD_UNSPECIFIED")
            HARM_BLOCK_THRESHOLD_UNSPECIFIED,
            /** Content with NEGLIGIBLE will be allowed. */
            @JsonProperty("BLOCK_LOW_AND_ABOVE")
            BLOCK_LOW_AND_ABOVE,
            /** Content with NEGLIGIBLE and LOW will be allowed. */
            @JsonProperty("BLOCK_MEDIUM_AND_ABOVE")
            BLOCK_MEDIUM_AND_ABOVE,

            /**
             * Content with NEGLIGIBLE, LOW, and MEDIUM will be allowed.
             */
            @JsonProperty("BLOCK_ONLY_HIGH")
            BLOCK_ONLY_HIGH,

            /**
             * All content will be allowed.
             */
            @JsonProperty("BLOCK_NONE")
            BLOCK_NONE
        }
    }

    /**
     * Configuration options for model generation and outputs. Not all parameters may be configurable
     * for every model.
     *
     * @param stopSequences Optional. The set of character sequences (up to 5) that will stop output
     *                      generation. If specified, the API will stop at the first appearance of a
     *                      stop sequence. The stop sequence will not be included as part of the response.
     * @param responseMimeType Optional. Output response mimetype of the generated candidate text. Supported
     *                         mimetypes: {@code text/plain} (default) for text output, {@code application/json}
     *                         for JSON response in the candidates.
     * @param responseSchema Optional. Output response schema of the generated candidate text when response
     *                       mime type can have schema. Schema can be objects, primitives or arrays and is a
     *                       subset of OpenAPI schema. If set, a compatible {@code responseMimeType} must also
     *                       be set. Compatible mimetypes: {@code application/json} for schema for JSON response.
     * @param candidateCount Optional. Number of generated responses to return. Currently, this value can only
     *                       be set to 1. If unset, this will default to 1.
     * @param maxOutputTokens Optional. The maximum number of tokens to include in a candidate. Note: The default
     *                        value varies by model, see the {@code Model.output_token_limit} attribute of the
     *                        {@code Model} returned from the {@code getModel} function.
     * @param temperature Optional. Controls the randomness of the output. Note: The default value varies by model,
     *                    see the {@code Model.temperature} attribute of the {@code Model} returned from the
     *                    {@code getModel} function. Values can range from [0.0, 2.0].
     * @param topP Optional. The maximum cumulative probability of tokens to consider when sampling. The model uses
     *             combined Top-k and nucleus sampling. Tokens are sorted based on their assigned probabilities so
     *             that only the most likely tokens are considered. Top-k sampling directly limits the maximum number
     *             of tokens to consider, while Nucleus sampling limits number of tokens based on the cumulative
     *             probability. Note: The default value varies by model, see the {@code Model.top_p} attribute of the
     *             {@code Model} returned from the {@code getModel} function.
     * @param topK Optional. The maximum number of tokens to consider when sampling. Models use nucleus sampling or
     *             combined Top-k and nucleus sampling. Top-k sampling considers the set of {@code topK} most probable
     *             tokens. Models running with nucleus sampling don't allow {@code topK} setting. Note: The default
     *             value varies by model, see the {@code Model.top_k} attribute of the {@code Model} returned from the
     *             {@code getModel} function. Empty {@code topK} field in {@code Model} indicates the model doesn't
     *             apply top-k sampling and doesn't allow setting {@code topK} on requests.
     */
    public record GenerationConfig(
            @JsonProperty("stopSequences") List<String> stopSequences,
            @JsonProperty("responseMimeType") String responseMimeType,
            @JsonProperty("responseSchema") Schema responseSchema,
            @JsonProperty("candidateCount") Integer candidateCount,
            @JsonProperty("maxOutputTokens") Integer maxOutputTokens,
            @JsonProperty("temperature") Double temperature,
            @JsonProperty("topP") Double topP,
            @JsonProperty("topK") Integer topK
    ) {
        /**
         * Creates a new builder for {@code GenerationConfig}.
         *
         * @return a new {@code Builder} instance.
         */
        public static Builder builder() {
            return new Builder();
        }

        /**
         * Builder class for {@code GenerationConfig}.
         */
        public static class Builder {
            private List<String> stopSequences;
            private String responseMimeType;
            private Schema responseSchema;
            private Integer candidateCount;
            private Integer maxOutputTokens;
            private Double temperature;
            private Double topP;
            private Integer topK;

            public Builder stopSequences(List<String> stopSequences) {
                this.stopSequences = stopSequences;
                return this;
            }

            public Builder responseMimeType(String responseMimeType) {
                this.responseMimeType = responseMimeType;
                return this;
            }

            public Builder responseSchema(Schema responseSchema) {
                this.responseSchema = responseSchema;
                return this;
            }

            public Builder candidateCount(Integer candidateCount) {
                this.candidateCount = candidateCount;
                return this;
            }

            public Builder maxOutputTokens(Integer maxOutputTokens) {
                this.maxOutputTokens = maxOutputTokens;
                return this;
            }

            public Builder temperature(Double temperature) {
                this.temperature = temperature;
                return this;
            }

            public Builder topP(Double topP) {
                this.topP = topP;
                return this;
            }

            public Builder topK(Integer topK) {
                this.topK = topK;
                return this;
            }

            public GenerationConfig build() {
                return new GenerationConfig(
                        stopSequences,
                        responseMimeType,
                        responseSchema,
                        candidateCount,
                        maxOutputTokens,
                        temperature,
                        topP,
                        topK
                );
            }
        }
    }

    /**
     * The Schema object allows the definition of input and output data types. These types can be objects,
     * but also primitives and arrays. Represents a select subset of an OpenAPI 3.0 schema object.
     *
     * @param type Required. Data type.
     * @param format Optional. The format of the data. This is used only for primitive datatypes. Supported formats:
     *               for {@code NUMBER} type: {@code float}, {@code double}; for {@code INTEGER} type: {@code int32}, {@code int64}.
     * @param description Optional. A brief description of the parameter. This could contain examples of use. Parameter
     *                    description may be formatted as Markdown.
     * @param nullable Optional. Indicates if the value may be null.
     * @param enumValues Optional. Possible values of the element of {@code Type.STRING} with enum format. For example,
     *                   we can define an Enum Direction as: {@code {type:STRING, format:enum, enum:["EAST", "NORTH", "SOUTH", "WEST"]}}.
     * @param properties Optional. Properties of {@code Type.OBJECT}. An object containing a list of "key": value pairs.
     *                   Example: {@code { "name": "wrench", "mass": "1.3kg", "count": "3" }}.
     * @param required Optional. Required properties of {@code Type.OBJECT}.
     * @param items Optional. Schema of the elements of {@code Type.ARRAY}.
     */
    public record Schema(
            @JsonProperty("type") Type type,
            @JsonProperty("format") String format,
            @JsonProperty("description") String description,
            @JsonProperty("nullable") Boolean nullable,
            @JsonProperty("enum") List<String> enumValues,
            @JsonProperty("properties") Map<String, Schema> properties,
            @JsonProperty("required") List<String> required,
            @JsonProperty("items") Schema items
    ) {
        /**
         * Enum representing the list of OpenAPI data types.
         */
        public enum Type {

            /** Not specified, should not be used. */
            @JsonProperty TYPE_UNSPECIFIED,

            /** String type. */
            @JsonProperty STRING,

            /** Number type. */
            @JsonProperty NUMBER,

            /** Integer type. */
            @JsonProperty INTEGER,

            /** Boolean type. */
            @JsonProperty BOOLEAN,

            /** Array type. */
            @JsonProperty ARRAY,

            /** Object type. */
            @JsonProperty OBJECT;

            @JsonCreator
            public static Type forValue(String name) {
                for (Type value : values()) {
                    if (value.name().equalsIgnoreCase(name)) {
                        return value;
                    }
                }
                throw new IllegalArgumentException("Unsupported Type value: " + name);
            }
        }
    }

    /**
     * Response from the model supporting multiple candidates.
     *
     * <p>Note on safety ratings and content filtering. They are reported for both prompt in
     * {@code GenerateContentResponse.promptFeedback} and for each candidate in {@code finishReason}
     * and in {@code safetyRatings}. The API contract is that:
     * <ul>
     *   <li>Either all requested candidates are returned or no candidates at all.</li>
     *   <li>No candidates are returned only if there was something wrong with the prompt (see {@code promptFeedback}).</li>
     *   <li>Feedback on each candidate is reported on {@code finishReason} and {@code safetyRatings}.</li>
     * </ul>
     *
     * @param candidates Candidate responses from the model.
     * @param promptFeedback Returns the prompt's feedback related to the content filters.
     * @param usageMetadata Output only. Metadata on the generation requests' token usage.
     */
    public record GenerateContentResponse(
            @JsonProperty("candidates") List<Candidate> candidates,
            @JsonProperty("promptFeedback") PromptFeedback promptFeedback,
            @JsonProperty("usageMetadata") UsageMetadata usageMetadata
    ) { }

    /**
     * A set of the feedback metadata for the prompt specified in {@code GenerateContentRequest.content}.
     *
     * @param blockReason Optional. If set, the prompt was blocked and no candidates are returned. Rephrase your prompt.
     * @param safetyRatings Ratings for safety of the prompt. There is at most one rating per category.
     */
    public record PromptFeedback(
            @JsonProperty("blockReason") BlockReason blockReason,
            @JsonProperty("safetyRatings") List<SafetyRating> safetyRatings
    ) {
        /**
         * Enum specifying the reason why a prompt was blocked.
         */
        public enum BlockReason {
            /** Default value. This value is unused. */
            @JsonProperty("BLOCK_REASON_UNSPECIFIED")
            BLOCK_REASON_UNSPECIFIED,

            /** Prompt was blocked due to safety reasons. You can inspect safetyRatings to understand which safety category blocked it. */
            @JsonProperty("SAFETY")
            SAFETY,

            /** Prompt was blocked due to unknown reasons. */
            @JsonProperty("OTHER")
            OTHER
        }
    }

    /**
     * Metadata on the generation request's token usage.
     *
     * @param promptTokenCount Number of tokens in the prompt.
     * @param candidatesTokenCount Total number of tokens across the generated candidates.
     * @param totalTokenCount Total token count for the generation request (prompt + candidates).
     */
    public record UsageMetadata(
            @JsonProperty("promptTokenCount") Integer promptTokenCount,
            @JsonProperty("candidatesTokenCount") Integer candidatesTokenCount,
            @JsonProperty("totalTokenCount") Integer totalTokenCount
    ) { }

    /**
     * Safety rating for a piece of content.
     *
     * <p>The safety rating contains the category of harm and the harm probability level in that category for a piece of content.
     * Content is classified for safety across a number of harm categories and the probability of the harm classification is included here.</p>
     *
     * @param category Required. The category for this rating.
     * @param probability Required. The probability of harm for this content.
     * @param blocked Was this content blocked because of this rating?
     */
    public record SafetyRating(
            @JsonProperty("category") SafetySetting.HarmCategory category,
            @JsonProperty("probability") HarmProbability probability,
            @JsonProperty("blocked") Boolean blocked
    ) {
        /**
         * Enum representing the probability that a piece of content is harmful.
         *
         * <p>The classification system gives the probability of the content being unsafe. This does not indicate the severity
         * of harm for a piece of content.</p>
         */
        public enum HarmProbability {
            /** Probability is unspecified. */
            @JsonProperty("HARM_PROBABILITY_UNSPECIFIED")
            HARM_PROBABILITY_UNSPECIFIED,

            /** Content has a negligible chance of being unsafe. */
            @JsonProperty("NEGLIGIBLE")
            NEGLIGIBLE,

            /** Content has a low chance of being unsafe. */
            @JsonProperty("LOW")
            LOW,

            /** Content has a medium chance of being unsafe. */
            @JsonProperty("MEDIUM")
            MEDIUM,

            /** Content has a high chance of being unsafe. */
            @JsonProperty("HIGH")
            HIGH
        }
    }

    /**
     * A response candidate generated from the model.
     *
     * @param content Output only. Generated content returned from the model.
     * @param finishReason Optional. Output only. The reason why the model stopped generating tokens.
     *                      If empty, the model has not stopped generating the tokens.
     * @param safetyRatings List of ratings for the safety of a response candidate. There is at most one rating per category.
     * @param citationMetadata Output only. Citation information for model-generated candidate. This field may be populated
     *                         with recitation information for any text included in the content. These are passages that are
     *                         "recited" from copyrighted material in the foundational LLM's training data.
     * @param tokenCount Output only. Token count for this candidate.
     * @param groundingAttributions Output only. Attribution information for sources that contributed to a grounded answer.
     *                               This field is populated for GenerateAnswer calls.
     * @param index Output only. Index of the candidate in the list of candidates.
     */
    public record Candidate(
            @JsonProperty("content") Content content,
            @JsonProperty("finishReason") FinishReason finishReason,
            @JsonProperty("safetyRatings") List<SafetyRating> safetyRatings,
            @JsonProperty("citationMetadata") CitationMetadata citationMetadata,
            @JsonProperty("tokenCount") Integer tokenCount,
            @JsonProperty("groundingAttributions") List<GroundingAttribution> groundingAttributions,
            @JsonProperty("index") Integer index
    ) {
        /**
         * Enum defining the reason why the model stopped generating tokens.
         */
        public enum FinishReason {
            /** Default value. This value is unused. */
            @JsonProperty("FINISH_REASON_UNSPECIFIED")
            FINISH_REASON_UNSPECIFIED,

            /** Natural stop point of the model or provided stop sequence. */
            @JsonProperty("STOP")
            STOP,

            /** The maximum number of tokens as specified in the request was reached. */
            @JsonProperty("MAX_TOKENS")
            MAX_TOKENS,

            /** The candidate content was flagged for safety reasons. */
            @JsonProperty("SAFETY")
            SAFETY,

            /** The candidate content was flagged for recitation reasons. */
            @JsonProperty("RECITATION")
            RECITATION,

            /** Unknown reason. */
            @JsonProperty("OTHER")
            OTHER
        }
    }

    /**
     * Attribution for a source that contributed to an answer.
     *
     * @param sourceId Output only. Identifier for the source contributing to this attribution.
     * @param content Grounding source content that makes up this attribution.
     */
    public record GroundingAttribution(
            @JsonProperty("sourceId") AttributionSourceId sourceId,
            @JsonProperty("content") Content content
    ) { }

    /**
     * Identifier for the source contributing to this attribution.
     *
     * @param groundingPassage Identifier for an inline passage.
     * @param semanticRetrieverChunk Identifier for a Chunk fetched via Semantic Retriever.
     */
    public record AttributionSourceId(
            @JsonProperty("groundingPassage") GroundingPassageId groundingPassage,
            @JsonProperty("semanticRetrieverChunk") SemanticRetrieverChunk semanticRetrieverChunk
    ) { }

    /**
     * Identifier for a part within a GroundingPassage.
     *
     * @param passageId Output only. ID of the passage matching the GenerateAnswerRequest's GroundingPassage.id.
     * @param partIndex Output only. Index of the part within the GenerateAnswerRequest's GroundingPassage.content.
     */
    public record GroundingPassageId(
            @JsonProperty("passageId") String passageId,
            @JsonProperty("partIndex") Integer partIndex
    ) { }

    /**
     * Identifier for a Chunk retrieved via Semantic Retriever specified in the GenerateAnswerRequest using SemanticRetrieverConfig.
     *
     * @param source Output only. Name of the source matching the request's SemanticRetrieverConfig.source. Example: corpora/123 or corpora/123/documents/abc.
     * @param chunk Output only. Name of the Chunk containing the attributed text. Example: corpora/123/documents/abc/chunks/xyz.
     */
    public record SemanticRetrieverChunk(
            @JsonProperty("source") String source,
            @JsonProperty("chunk") String chunk
    ) { }

    /**
     * A collection of source attributions for a piece of content.
     *
     * @param citationSources Citations to sources for a specific response.
     */
    public record CitationMetadata(
            @JsonProperty("citationSources") List<CitationSource> citationSources
    ) { }

    /**
     * A citation to a source for a portion of a specific response.
     *
     * @param startIndex Optional. Start of segment of the response that is attributed to this source.
     *                   Index indicates the start of the segment, measured in bytes.
     * @param endIndex Optional. End of the attributed segment, exclusive.
     * @param uri Optional. URI that is attributed as a source for a portion of the text.
     * @param license Optional. License for the GitHub project that is attributed as a source for segment.
     *                License info is required for code citations.
     */
    public record CitationSource(
            @JsonProperty("startIndex") Integer startIndex,
            @JsonProperty("endIndex") Integer endIndex,
            @JsonProperty("uri") String uri,
            @JsonProperty("license") String license
    ) { }

    public ResponseEntity<GenerateContentResponse> generateContent(String modelName, GenerateContentRequest request) {
        Assert.notNull(request, "The request body can not be null.");

        return this.restClient.post()
                .uri("/v1beta/models/" + URLEncoder.encode(modelName, StandardCharsets.UTF_8) + ":generateContent")
                .body(request)
                .retrieve()
                .toEntity(GenerateContentResponse.class);
    }

    /**
     * Creates a streaming chat response for the given chat conversation.
     *
     * @param modelName the name of the model to use
     * @param request the chat completion request, must have the stream property set to true
     * @return a {@link Flux} stream of {@link GenerateContentResponse} from chat completion chunks
     */
    public Flux<GenerateContentResponse> streamGenerateContent(String modelName, GenerateContentRequest request) {
        Assert.notNull(request, "The request can not be null.");

        AtomicBoolean isFunctionCall = new AtomicBoolean(false);

        return this.webClient.post()
                .uri("/v1beta/models/" + URLEncoder.encode(modelName, StandardCharsets.UTF_8) + ":streamGenerateContent?alt=sse")
                .body(Mono.just(request), GenerateContentRequest.class)
                .retrieve()
                .bodyToFlux(String.class)
                .map(content -> ModelOptionsUtils.jsonToObject(content, GenerateContentResponse.class))
                .map(chunk -> {
                    Content c;
                    final var functionCall = (c = chunk.candidates().get(0).content()) == null
                            ? null
                            : c.parts().get(0).functionCall();
                    isFunctionCall.set(functionCall != null && !functionCall.name().isEmpty());
                    return chunk;
                })
                .windowUntil(chunk -> {
                    if (isFunctionCall.get() && chunk.candidates()
                            .get(0)
                            .finishReason() == Candidate.FinishReason.STOP) {
                        isFunctionCall.set(false);
                        return true;
                    }
                    return !isFunctionCall.get();
                })
                .concatMapIterable(window -> {
                    return List.of(window);
                    //final var reduce = window.reduce(MergeUtils.emptyChatCompletions(), MergeUtils::mergeChatCompletions);
                    //return List.of(reduce);
                })
                .flatMap(mono -> mono);
    }

}
