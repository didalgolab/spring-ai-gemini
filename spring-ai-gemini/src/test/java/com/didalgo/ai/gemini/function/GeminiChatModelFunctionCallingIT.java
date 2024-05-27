package com.didalgo.ai.gemini.function;

import com.didalgo.ai.gemini.GeminiChatModel;
import com.didalgo.ai.gemini.GeminiChatOptions;
import com.didalgo.ai.gemini.MessageAggregatorExt;
import com.didalgo.ai.gemini.api.GeminiApi;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.MessageAggregator;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.function.FunctionCallbackWrapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Bean;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@EnabledIfEnvironmentVariable(named = "GEMINI_API_KEY", matches = ".*")
public class GeminiChatModelFunctionCallingIT {

    private final Logger logger = LoggerFactory.getLogger(getClass());

    private static final MessageAggregator aggregator = new MessageAggregatorExt();

    @Autowired
    private GeminiChatModel geminiModel;

    @Test
    void testFindTheaters() {
        UserMessage userMessage = new UserMessage("Which theaters in Mountain View show Barbie movie?");

        GeminiChatOptions promptOptions = GeminiChatOptions.builder()
                .withFunctions(Set.of("find_movies", "find_theaters", "get_showtimes"))
                .build();

        ChatResponse response = geminiModel.call(new Prompt(userMessage, promptOptions));
        logger.info("Response: {}", response);

        assertThat(response.getResult().getOutput().getContent())
                .contains("Mountain View Center for the Performing Arts", "Mountain View High School Theater", "Guild Theatre of Mountain View");
    }

    @Test
    void testFindTheatersStreaming() {
        UserMessage userMessage = new UserMessage("Which theaters in Mountain View show Barbie movie?");

        GeminiChatOptions promptOptions = GeminiChatOptions.builder()
                .withFunctions(Set.of("find_movies", "find_theaters", "get_showtimes"))
                .build();

        Flux<ChatResponse> stream = geminiModel.stream(new Prompt(userMessage, promptOptions));
        AtomicReference<ChatResponse> response = new AtomicReference<>();
        aggregator.aggregate(stream, response::set).blockLast();
        logger.info("Response: {}", response.get());

        assertThat(response.get().getResult().getOutput().getContent())
                .contains("Mountain View Center for the Performing Arts", "Mountain View High School Theater", "Guild Theatre of Mountain View");
    }

    @Test
    void testMultiTurnFunctionCalling() {
        GeminiChatOptions promptOptions = GeminiChatOptions.builder()
                .withFunctions(Set.of("find_movies", "find_theaters", "get_showtimes"))
                .build();

        // Turn 1
        UserMessage userMessage = new UserMessage("Which theaters in Mountain View show Barbie movie?");
        ChatResponse response = geminiModel.call(new Prompt(userMessage, promptOptions));
        logger.info("Response: {}", response);

        assertThat(response.getResult().getOutput().getContent())
                .contains("Mountain View Center for the Performing Arts", "Mountain View High School Theater", "Guild Theatre of Mountain View");

        // Turn 2
        userMessage = new UserMessage("Can we recommend some comedy movies on show in Mountain View?");
        response = geminiModel.call(new Prompt(userMessage, promptOptions));
        logger.info("Response: {}", response);

        assertThat(response.getResult().getOutput().getContent())
                .contains("Movie 1", "Movie 2", "Movie 3");
    }

    @Test
    void testMultiTurnFunctionCallingStreaming() {
        GeminiChatOptions promptOptions = GeminiChatOptions.builder()
                .withFunctions(Set.of("find_movies", "find_theaters", "get_showtimes"))
                .build();
        AtomicReference<ChatResponse> response = new AtomicReference<>();

        // Turn 1
        UserMessage userMessage = new UserMessage("Which theaters in Mountain View show Barbie movie?");
        aggregator.aggregate(
                geminiModel.stream(
                        new Prompt(userMessage, promptOptions)), response::set).blockLast();

        logger.info("Response: {}", response.get());
        assertThat(response.get().getResult().getOutput().getContent())
                .contains("Mountain View Center for the Performing Arts", "Mountain View High School Theater", "Guild Theatre of Mountain View");

        // Turn 2
        userMessage = new UserMessage("Can we recommend some comedy movies on show in Mountain View?");
        aggregator.aggregate(
                geminiModel.stream(
                        new Prompt(userMessage, promptOptions)), response::set).blockLast();

        logger.info("Response: {}", response.get());
        assertThat(response.get().getResult().getOutput().getContent())
                .contains("Movie 1", "Movie 2", "Movie 3");
    }

    @Test
    void testMultipleFunctionCalling() {
        GeminiChatOptions promptOptions = GeminiChatOptions.builder()
                .withModel("gemini-1.5-flash-latest")
                .withTemperature(0f)
                .withFunctions(Set.of("get_current_weather"))
                .build();

        UserMessage userMessage = new UserMessage("What is the current difference in temperature in New Delhi and San Francisco? Hint: You may query weather information multiple times.");
        //UserMessage userMessage = new UserMessage("What is difference in temperature in New Delhi and San Francisco?");
        ChatResponse response = geminiModel.call(new Prompt(userMessage, promptOptions));
        logger.info("Response: {}", response);

        assertThat(response.getResult().getOutput().getContent()).containsAnyOf("10.5", "10,5");
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
                    .withTemperature(0.5f)
                    .withFunctionCallbacks(List.of(
                            FunctionCallbackWrapper.builder(new MockMovieService.FindMoviesFunction())
                                    .withName("find_movies")
                                    .withDescription("find movie titles currently playing in theaters based on any description, genre, title words, etc.")
                                    .build(),
                            FunctionCallbackWrapper.builder(new MockMovieService.FindTheatersFunction())
                                    .withName("find_theaters")
                                    .withDescription("find theaters based on location and optionally movie title which is currently playing in theaters")
                                    .build(),
                            FunctionCallbackWrapper.builder(new MockMovieService.GetShowtimesFunction())
                                    .withName("get_showtimes")
                                    .withDescription("Find the start times for movies playing in a specific theater")
                                    .build(),
                            FunctionCallbackWrapper.builder(new MockWeatherService())
                                    .withName("get_current_weather")
                                    .withDescription("Get the current weather in a specific location")
                                    .build()))
                    .build();
        }

        @Bean
        public GeminiChatModel geminiChatClient(GeminiApi api, GeminiChatOptions options) {
            return new GeminiChatModel(api, options);
        }
    }
}
