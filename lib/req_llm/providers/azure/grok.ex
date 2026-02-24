defmodule ReqLLM.Providers.Azure.Grok do
  @moduledoc """
  xAI Grok model family support for Azure AI services.

  Handles Grok models (grok-3, grok-4, etc.) deployed on Azure's infrastructure.

  This module acts as a thin adapter between Azure's deployment-based API
  and xAI's OpenAI-compatible Chat Completions format, delegating encoding to
  `ReqLLM.Provider.Defaults` and applying Grok-specific modifications.

  ## Key Differences from Native xAI

  - No `model` field in request body (Azure deployment determines the model)
  - Uses `api-key` header instead of xAI's `Authorization: Bearer` header
  - Endpoint paths follow Azure conventions (deployment-based or Foundry)

  ## Grok-Specific Extensions

  Beyond standard OpenAI parameters, Grok models support:
  - `max_completion_tokens` - Preferred over max_tokens for Grok-4 models
  - `reasoning_effort` - Reasoning level (low, medium, high) for grok-3-mini models only

  ## Model Compatibility Notes

  - `reasoning_effort` is only supported for grok-3-mini and grok-3-mini-fast models
  - Grok-4 models do not support `stop`, `presence_penalty`, or `frequency_penalty`
  """

  import ReqLLM.Provider.Utils, only: [maybe_put: 3]

  alias ReqLLM.Provider.Defaults

  require Logger

  @anthropic_specific_options [
    :anthropic_prompt_cache,
    :anthropic_prompt_cache_ttl,
    :anthropic_version
  ]

  @openai_specific_options [
    :service_tier,
    :verbosity,
    :openai_structured_output_mode,
    :openai_parallel_tool_calls
  ]

  @doc """
  Pre-validates and transforms options for Grok models on Azure.
  Warns if Anthropic-specific or OpenAI-specific options are passed.
  """
  def pre_validate_options(_operation, _model, opts) do
    opts
    |> warn_and_remove_incompatible_options(@anthropic_specific_options, "Anthropic")
    |> warn_and_remove_incompatible_options(@openai_specific_options, "OpenAI")
    |> then(&{&1, []})
  end

  defp warn_and_remove_incompatible_options(opts, option_keys, provider_name) do
    case opts[:provider_options] do
      provider_opts when is_list(provider_opts) ->
        found = Enum.filter(option_keys, &Keyword.has_key?(provider_opts, &1))

        if found == [] do
          opts
        else
          Logger.warning(
            "Options #{inspect(found)} are #{provider_name}-specific and are ignored for Grok models on Azure."
          )

          updated = Keyword.drop(provider_opts, found)
          Keyword.put(opts, :provider_options, updated)
        end

      _ ->
        opts
    end
  end

  @doc """
  Formats a ReqLLM context into OpenAI Chat Completions request format for Grok.

  Delegates encoding to `ReqLLM.Provider.Defaults.default_build_body/1` then
  applies Grok-specific modifications:
  - Removes `model` field (Azure uses deployment-based routing)
  - Adds `max_completion_tokens` (preferred over `max_tokens` for Grok-4)
  - Adds `reasoning_effort` for grok-3-mini models

  Returns a map ready to be JSON-encoded for the Azure API.
  """
  def format_request(model_id, context, opts) do
    provider_opts = opts[:provider_options] || []

    temp_request =
      Req.new(method: :post, url: URI.parse("https://example.com/temp"))
      |> Map.put(:body, {:json, %{}})
      |> Map.put(
        :options,
        Map.new(
          [
            model: model_id,
            context: context,
            operation: opts[:operation] || :chat,
            tools: opts[:tools] || []
          ] ++ Keyword.drop(opts, [:model, :tools, :operation, :provider_options])
        )
      )

    body = Defaults.default_build_body(temp_request)

    body
    |> Map.drop([:model, "model"])
    |> maybe_put(:max_completion_tokens, opts[:max_completion_tokens] || opts[:max_tokens])
    |> Map.delete(:max_tokens)
    |> maybe_put(:reasoning_effort, normalize_reasoning_effort(provider_opts[:reasoning_effort]))
    |> add_stream_options(opts)
  end

  defp normalize_reasoning_effort(nil), do: nil
  defp normalize_reasoning_effort(v) when is_atom(v), do: Atom.to_string(v)
  defp normalize_reasoning_effort(v) when is_binary(v), do: v

  defp add_stream_options(body, opts) do
    if opts[:stream] do
      maybe_put(body, :stream_options, %{include_usage: true})
    else
      body
    end
  end

  @doc """
  Grok models do not support embeddings.
  """
  def format_embedding_request(_model_id, _text, _opts) do
    {:error,
     ReqLLM.Error.Invalid.Parameter.exception(
       parameter: "Grok models do not support embeddings. Use an OpenAI embedding model."
     )}
  end

  @doc """
  Parses an Azure Grok response into ReqLLM format.

  Uses the centralized OpenAI response decoding from Provider.Defaults.
  """
  def parse_response(body, model, opts) do
    context = opts[:context] || %ReqLLM.Context{messages: []}
    operation = opts[:operation]

    {:ok, response} = Defaults.decode_response_body_openai_format(body, model)

    merged_response = ReqLLM.Context.merge_response(context, response)

    final_response =
      if operation == :object do
        extract_and_set_object(merged_response)
      else
        merged_response
      end

    {:ok, final_response}
  end

  defp extract_and_set_object(response) do
    extracted_object =
      response
      |> ReqLLM.Response.tool_calls()
      |> ReqLLM.ToolCall.find_args("structured_output")

    %{response | object: extracted_object}
  end

  @doc """
  Extracts usage information from Azure Grok response.

  Includes all available fields: input_tokens, output_tokens, total_tokens,
  and reasoning_tokens (from completion_tokens_details).
  """
  def extract_usage(body, _model) when is_map(body) do
    case body do
      %{"usage" => usage} ->
        reasoning =
          get_in(usage, ["completion_tokens_details", "reasoning_tokens"]) || 0

        {:ok,
         %{
           input_tokens: Map.get(usage, "prompt_tokens", 0),
           output_tokens: Map.get(usage, "completion_tokens", 0),
           total_tokens: Map.get(usage, "total_tokens", 0),
           cached_tokens: 0,
           reasoning_tokens: reasoning
         }}

      _ ->
        {:error, :no_usage}
    end
  end

  def extract_usage(_, _), do: {:error, :no_usage}

  @doc """
  Decodes Server-Sent Events for streaming responses.

  Uses the same SSE format as standard OpenAI.
  """
  def decode_stream_event(event, model) do
    ReqLLM.Provider.Defaults.default_decode_stream_event(event, model)
  end
end
