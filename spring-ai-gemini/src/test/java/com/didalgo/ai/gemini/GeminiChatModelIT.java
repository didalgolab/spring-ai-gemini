package com.didalgo.ai.gemini;

import com.didalgo.ai.gemini.api.GeminiApi;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
public class GeminiChatModelIT {

    private final Logger logger = LoggerFactory.getLogger(getClass());

    private static final MessageAggregator aggregator = new MessageAggregatorExt();

    @Autowired
    private GeminiChatModel geminiModel;

    @Test
    void withSafetySettings_can_answer_innocuous_question() {
        UserMessage userMessage = new UserMessage("The largest prime number lower than 10000?");

        GeminiChatOptions promptOptions = GeminiChatOptions.builder()
                .withSafetySettings(GeminiApi.SafetySettings.BLOCK_NONE)
                .build();

        ChatResponse response = geminiModel.call(new Prompt(userMessage, promptOptions));
        logger.info("Response: {}", response);

        assertThat(response.getResult().getOutput().getContent())
                .containsAnyOf("9973", "9,973");
    }

    @SpringBootConfiguration
    public static class TestConfiguration {

        @Bean
        public GeminiApi geminiApi() {
            String apiKey = System.getenv("GEMINI_API_KEY");
            return new GeminiApi(Optional.ofNullable(apiKey).orElseThrow(() -> new AssertionError("Missing Gemini API Key")));
        }

        @Bean
        public GeminiChatOptions geminiChatOptions() {
            return GeminiChatOptions.builder()
                    .withModel("gemini-1.5-flash-latest")
                    .withTemperature(0.5)
                    .build();
        }

        @Bean
        public GeminiChatModel geminiChatClient(GeminiApi api, GeminiChatOptions options) {
            return new GeminiChatModel(api, options);
        }
    }
}
