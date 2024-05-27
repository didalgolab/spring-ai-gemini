package com.didalgo.ai.gemini.function;

import com.fasterxml.jackson.annotation.JsonClassDescription;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.function.Function;

public class MockWeatherService implements Function<MockWeatherService.Request, MockWeatherService.Response> {

	private final Logger logger = LoggerFactory.getLogger(getClass());

	/**
	 * Weather Function request.
	 */
	@JsonInclude(Include.NON_NULL)
	@JsonClassDescription("Weather API request")
	public record Request(
			@JsonProperty(required = true, value = "location")
			@JsonPropertyDescription("The city and state, e.g. San Francisco, CA or a zip code e.g. 95616")
			String location,

			@JsonProperty(required = false, value = "unit")
			@JsonPropertyDescription("Temperature unit")
			Unit unit) {
	}

	/**
	 * Temperature units.
	 */
	public enum Unit {

		/**
		 * Celsius.
		 */
		C("metric"),
		/**
		 * Fahrenheit.
		 */
		F("imperial");

		/**
		 * Human readable unit name.
		 */
		public final String unitName;

		private Unit(String text) {
			this.unitName = text;
		}

	}

	/**
	 * Weather Function response.
	 */
	public record Response(double temp, Unit unit) {
	}

	@Override
	public Response apply(Request request) {

		double temperature = 0;
		if (request.location().contains("New Delhi")) {
			temperature = 30.5;
		}
		else if (request.location().contains("Paris")) {
			temperature = 15;
		}
		else if (request.location().contains("San Francisco")) {
			temperature = 20;
		}
		else if (request.location().contains("Tokyo")) {
			temperature = 10;
		}

		logger.info("Request is {}, response temperature is {}", request, temperature);
		return new Response(temperature, Unit.C);
	}

}