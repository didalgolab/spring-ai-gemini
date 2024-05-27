package com.didalgo.ai.gemini.function;

import java.util.List;
import java.util.function.Function;

import com.fasterxml.jackson.annotation.JsonClassDescription;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MockMovieService {

    private static final Logger logger = LoggerFactory.getLogger(MockMovieService.class);

    @JsonInclude(Include.NON_NULL)
    @JsonClassDescription("Find movie titles currently playing in theaters based on any description, genre, title words, etc.")
    public record FindMoviesRequest(
        @JsonProperty(required = false, value = "location")
        @JsonPropertyDescription("The city and state, e.g. San Francisco, CA or a zip code e.g. 95616")
        String location,

        @JsonProperty(required = true, value = "description")
        @JsonPropertyDescription("Any kind of description including category or genre, title words, attributes, etc.")
        String description
    ) {}

    @JsonInclude(Include.NON_NULL)
    @JsonClassDescription("Find theaters based on location and optionally movie title which is currently playing in theaters")
    public record FindTheatersRequest(
        @JsonProperty(required = true, value = "location")
        @JsonPropertyDescription("The city and state, e.g. San Francisco, CA or a zip code e.g. 95616")
        String location,

        @JsonProperty(required = false, value = "movie")
        @JsonPropertyDescription("Any movie title")
        String movie
    ) {}

    @JsonInclude(Include.NON_NULL)
    @JsonClassDescription("Find the start times for movies playing in a specific theater")
    public record GetShowtimesRequest(
        @JsonProperty(required = true, value = "location")
        @JsonPropertyDescription("The city and state, e.g. San Francisco, CA or a zip code e.g. 95616")
        String location,

        @JsonProperty(required = true, value = "movie")
        @JsonPropertyDescription("Any movie title")
        String movie,

        @JsonProperty(required = true, value = "theater")
        @JsonPropertyDescription("Name of the theater")
        String theater,

        @JsonProperty(required = true, value = "date")
        @JsonPropertyDescription("Date for requested showtime")
        String date
    ) {}

    @JsonInclude(Include.NON_NULL)
    @JsonClassDescription("Response with a list of movie titles")
    public record FindMoviesResponse(
        @JsonProperty("movies") List<String> movies
    ) {}

    @JsonInclude(Include.NON_NULL)
    public record FoundTheaters(
            @JsonProperty("movie") String movie,
            @JsonProperty("theaters") List<Theater> theaters
    ) {}

    @JsonInclude(Include.NON_NULL)
    public record Theater(
            @JsonProperty("name") String name,
            @JsonProperty("address") String address
    ) {}

    @JsonInclude(Include.NON_NULL)
    @JsonClassDescription("Response with a list of showtimes")
    public record GetShowtimesResponse(
        @JsonProperty("showtimes") List<String> showtimes
    ) {}

    public static class FindMoviesFunction implements Function<FindMoviesRequest, FindMoviesResponse> {
        @Override
        public FindMoviesResponse apply(FindMoviesRequest request) {
            System.out.println("FindMoviesRequest received: " + request);
            if (request.location.contains("Mountain View")) {

            }
            return new FindMoviesResponse(List.of("Movie 1", "Movie 2", "Movie 3"));
        }
    };

    public static class FindTheatersFunction implements Function<FindTheatersRequest, FoundTheaters> {
        @Override
        public FoundTheaters apply(FindTheatersRequest request) {
            logger.info("FindTheatersRequest received: {}", request);
            if (request.movie().equals("Barbie") && request.location().contains("Mountain View")) {
                var location = "Mountain View";
                return new FoundTheaters("Barbie", List.of(
                        new Theater(location + " Center for the Performing Arts", ""),
                        new Theater(location + " High School Theater", ""),
                        new Theater("Guild Theatre of " + location, "")));
            }
            return new FoundTheaters(request.movie(), List.of());
        }
    }

    public static class GetShowtimesFunction implements Function<GetShowtimesRequest, GetShowtimesResponse> {
        @Override
        public GetShowtimesResponse apply(GetShowtimesRequest request) {
            logger.info("GetShowtimesRequest received: {}", request);
            return new GetShowtimesResponse(List.of("10:00 AM", "12:00 PM", "2:00 PM"));
        }
    }
}
