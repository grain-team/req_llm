defmodule ReqLLM.Message do
  @moduledoc """
  Message represents a single conversation message with multi-modal content support.

  Content is always a list of `ContentPart` structs, never a string.
  This ensures consistent handling across all providers and eliminates polymorphism.

  ## Reasoning Details

  The `reasoning_details` field contains provider-specific reasoning metadata that must
  be preserved across conversation turns for reasoning models. This field is:
  - `nil` for non-reasoning models or models that don't provide structured reasoning metadata
  - A list of normalized ReasoningDetails for reasoning models

  For multi-turn reasoning continuity, include the previous assistant message
  (with its reasoning_details) in subsequent requests.
  """

  use TypedStruct

  alias ReqLLM.Message.ContentPart
  alias ReqLLM.ToolCall

  typedstruct enforce: false, module: ReasoningDetails do
    @moduledoc """
    Normalized reasoning/thinking data from LLM providers.

    ## Fields
    - `text` - Human-readable reasoning/thinking text (may be summarized)
    - `signature` - Opaque signature/token for multi-turn continuity
    - `encrypted?` - Whether the signature contains encrypted reasoning tokens
    - `provider` - Source provider (:anthropic, :google, :openai, :openrouter)
    - `format` - Provider-specific format version identifier
    - `index` - Position index for ordered reasoning blocks
    - `provider_data` - Raw provider-specific fields for lossless round-trips
    """
    @derive Jason.Encoder
    field(:text, String.t())
    field(:signature, String.t())
    field(:encrypted?, boolean(), default: false)
    field(:provider, atom())
    field(:format, String.t())
    field(:index, non_neg_integer(), default: 0)
    field(:provider_data, map(), default: %{})
  end

  @derive Jason.Encoder
  typedstruct enforce: true do
    field(:role, :user | :assistant | :system | :tool, enforce: true)
    field(:content, [ContentPart.t()], default: [])
    field(:name, String.t() | nil, default: nil)
    field(:tool_call_id, String.t() | nil, default: nil)
    field(:tool_calls, [ToolCall.t()] | nil, default: nil)
    field(:metadata, map(), default: %{})
    field(:reasoning_details, [ReasoningDetails.t()] | nil, default: nil)
  end

  @spec valid?(t()) :: boolean()
  def valid?(%__MODULE__{content: content}) when is_list(content), do: true
  def valid?(_), do: false

  defimpl Inspect do
    def inspect(%{role: role, content: parts}, opts) do
      summary =
        parts
        |> Enum.map_join(",", & &1.type)

      Inspect.Algebra.concat(["#Message<", Inspect.Algebra.to_doc(role, opts), " ", summary, ">"])
    end
  end
end
