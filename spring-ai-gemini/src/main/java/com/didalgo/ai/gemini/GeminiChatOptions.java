package com.didalgo.ai.gemini;

import com.didalgo.ai.gemini.api.GeminiApi.FunctionCallingMode;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallingOptions;
import org.springframework.boot.context.properties.NestedConfigurationProperty;
import org.springframework.util.Assert;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@JsonInclude(JsonInclude.Include.NON_NULL)
public class GeminiChatOptions
        implements FunctionCallingOptions, ChatOptions {

    /**
     * ID of the model to use.
     */
    private @JsonIgnore String model;

    /**
     * Optional. Stop sequences.
     */
    private @JsonProperty("stopSequences") List<String> stopSequences;

    /**
     * Optional. Controls the randomness of predictions.
     */
    private @JsonProperty("temperature") Float temperature;

    /**
     * Optional. If specified, nucleus sampling will be used.
     */
    private @JsonProperty("topP") Float topP;

    /**
     * Optional. If specified, top k sampling will be used.
     */
    private @JsonProperty("topK") Integer topK;

    /**
     * Optional. The maximum number of tokens to generate.
     */
    private @JsonProperty("candidateCount") Integer candidateCount;

    /**
     * Optional. The maximum number of tokens to generate.
     */
    private @JsonProperty("maxOutputTokens") Integer maxOutputTokens;

    /**
     * Optional. The mode in which function calling should execute. Defaults to {@link FunctionCallingMode#AUTO}.
     */
    private @JsonProperty("functionCallingMode") FunctionCallingMode functionCallingMode;

    /**
     * Tool Function Callbacks to register with the ChatClient.
     * For Prompt Options the functionCallbacks are automatically enabled for the duration of the prompt execution.
     * For Default Options the functionCallbacks are registered but disabled by default. Use the enableFunctions to set the functions
     * from the registry to be used by the ChatClient chat completion requests.
     */
    @NestedConfigurationProperty
    @JsonIgnore
    private List<FunctionCallback> functionCallbacks = new ArrayList<>();

    /**
     * List of functions, identified by their names, to configure for function calling in
     * the chat completion requests.
     * Functions with those names must exist in the functionCallbacks registry.
     * The {@link #functionCallbacks} from the PromptOptions are automatically enabled for the duration of the prompt execution.
     * <p>
     * Note that function enabled with the default options are enabled for all chat completion requests. This could impact the token count and the billing.
     * If the functions is set in a prompt options, then the enabled functions are only active for the duration of this prompt execution.
     */
    @NestedConfigurationProperty
    @JsonIgnore
    private Set<String> functions = new HashSet<>();


    public static GeminiChatOptions.Builder builder() {
        return new GeminiChatOptions.Builder();
    }

    public String getModel() {
        return model;
    }

    @Override
    public Float getTemperature() {
        return temperature;
    }

    @Override
    public Float getTopP() {
        return topP;
    }

    @Override
    public Integer getTopK() {
        return topK;
    }

    public Integer getCandidateCount() {
        return candidateCount;
    }

    public Integer getMaxOutputTokens() {
        return maxOutputTokens;
    }

    public List<String> getStopSequences() {
        return stopSequences;
    }

    public FunctionCallingMode getFunctionCallingMode() {
        return functionCallingMode;
    }

    @Override
    public List<FunctionCallback> getFunctionCallbacks() {
        return functionCallbacks;
    }

    @Override
    public void setFunctionCallbacks(List<FunctionCallback> functionCallbacks) {
        this.functionCallbacks = functionCallbacks;
    }

    @Override
    public Set<String> getFunctions() {
        return functions;
    }

    @Override
    public void setFunctions(Set<String> functions) {
        this.functions = functions;
    }

    public static class Builder {
        protected GeminiChatOptions options;

        public Builder() {
            this.options = new GeminiChatOptions();
        }

        public Builder(GeminiChatOptions options) {
            this.options = options;
        }

        public Builder withModel(String model) {
            this.options.model = model;
            return this;
        }

        public Builder withTemperature(Float temperature) {
            this.options.temperature = temperature;
            return this;
        }

        public Builder withTopP(Float topP) {
            this.options.topP = topP;
            return this;
        }

        public Builder withTopK(Integer topK) {
            this.options.topK = topK;
            return this;
        }

        public Builder withCandidateCount(Integer candidateCount) {
            this.options.candidateCount = candidateCount;
            return this;
        }

        public Builder withMaxOutputTokens(Integer maxOutputTokens) {
            this.options.maxOutputTokens = maxOutputTokens;
            return this;
        }

        public Builder withStopSequences(List<String> stopSequences) {
            this.options.stopSequences = stopSequences;
            return this;
        }

        public Builder withFunctionCallbacks(List<FunctionCallback> functionCallbacks) {
            this.options.functionCallbacks = functionCallbacks;
            return this;
        }

        public Builder withFunctions(Set<String> functionNames) {
            Assert.notNull(functionNames, "Function names must not be null");
            this.options.functions = functionNames;
            return this;
        }

        public Builder withFunction(String functionName) {
            Assert.hasText(functionName, "Function name must not be empty");
            this.options.functions.add(functionName);
            return this;
        }

        public Builder withFunctionCallingMode(FunctionCallingMode mode) {
            this.options.functionCallingMode = mode;
            return this;
        }

        public GeminiChatOptions build() {
            return this.options;
        }
    }

    public static GeminiChatOptions fromOptions(GeminiChatOptions fromOptions) {
        return GeminiChatOptions.builder()
                .withModel(fromOptions.getModel())
                .withTemperature(fromOptions.getTemperature())
                .withTopP(fromOptions.getTopP())
                .withTopK(fromOptions.getTopK())
                .withCandidateCount(fromOptions.getCandidateCount())
                .withMaxOutputTokens(fromOptions.getMaxOutputTokens())
                .withStopSequences(fromOptions.getStopSequences())
                .withFunctionCallingMode(fromOptions.getFunctionCallingMode())
                .withFunctionCallbacks(fromOptions.getFunctionCallbacks())
                .withFunctions(fromOptions.getFunctions())
                .build();
    }
}