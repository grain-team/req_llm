defmodule ReqLLM.ReqPluginsTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Generation, Embedding, Response}

  @chat_response %{
    "id" => "cmpl_plugins_test",
    "model" => "gpt-4o-mini-2024-07-18",
    "choices" => [
      %{
        "message" => %{"role" => "assistant", "content" => "Hello from plugins test!"}
      }
    ],
    "usage" => %{"prompt_tokens" => 10, "completion_tokens" => 8, "total_tokens" => 18}
  }

  @embedding_response %{
    "object" => "list",
    "data" => [%{"object" => "embedding", "index" => 0, "embedding" => [0.1, 0.2, 0.3]}],
    "model" => "text-embedding-3-small",
    "usage" => %{"prompt_tokens" => 5, "total_tokens" => 5}
  }

  @object_response %{
    "id" => "cmpl_object_test",
    "model" => "gpt-4o-mini-2024-07-18",
    "choices" => [
      %{
        "message" => %{
          "role" => "assistant",
          "content" => nil,
          "tool_calls" => [
            %{
              "id" => "call_1",
              "type" => "function",
              "function" => %{
                "name" => "structured_output",
                "arguments" => ~s({"name":"Alice","age":30})
              }
            }
          ]
        }
      }
    ],
    "usage" => %{"prompt_tokens" => 20, "completion_tokens" => 15, "total_tokens" => 35}
  }

  describe "generate_text/3 with :req_plugins" do
    test "single plugin adds a custom header" do
      test_pid = self()

      Req.Test.stub(ReqLLM.ReqPluginsTest.SingleHeader, fn conn ->
        send(test_pid, {:headers, conn.req_headers})
        Req.Test.json(conn, @chat_response)
      end)

      plugin = fn req ->
        Req.Request.put_header(req, "x-custom-plugin", "test-value")
      end

      {:ok, response} =
        Generation.generate_text(
          "openai:gpt-4o-mini",
          "Hello",
          req_plugins: [plugin],
          req_http_options: [plug: {Req.Test, ReqLLM.ReqPluginsTest.SingleHeader}]
        )

      assert %Response{} = response
      assert_receive {:headers, headers}
      assert List.keyfind(headers, "x-custom-plugin", 0) == {"x-custom-plugin", "test-value"}
    end

    test "multiple plugins applied in order" do
      test_pid = self()

      Req.Test.stub(ReqLLM.ReqPluginsTest.MultiPlugins, fn conn ->
        send(test_pid, {:headers, conn.req_headers})
        Req.Test.json(conn, @chat_response)
      end)

      plugin_a = fn req ->
        Req.Request.put_header(req, "x-order", "first")
      end

      plugin_b = fn req ->
        Req.Request.put_header(req, "x-order", "second")
      end

      {:ok, _response} =
        Generation.generate_text(
          "openai:gpt-4o-mini",
          "Hello",
          req_plugins: [plugin_a, plugin_b],
          req_http_options: [plug: {Req.Test, ReqLLM.ReqPluginsTest.MultiPlugins}]
        )

      assert_receive {:headers, headers}
      assert List.keyfind(headers, "x-order", 0) == {"x-order", "second"}
    end

    test "empty list is a no-op" do
      Req.Test.stub(ReqLLM.ReqPluginsTest.EmptyPlugins, fn conn ->
        Req.Test.json(conn, @chat_response)
      end)

      {:ok, response} =
        Generation.generate_text(
          "openai:gpt-4o-mini",
          "Hello",
          req_plugins: [],
          req_http_options: [plug: {Req.Test, ReqLLM.ReqPluginsTest.EmptyPlugins}]
        )

      assert %Response{} = response
    end

    test "crashing plugin returns {:error, ...} with message" do
      crashing_plugin = fn _req ->
        raise "plugin exploded"
      end

      {:error, error} =
        Generation.generate_text(
          "openai:gpt-4o-mini",
          "Hello",
          req_plugins: [crashing_plugin],
          req_http_options: [plug: {Req.Test, __MODULE__}]
        )

      assert Exception.message(error) =~ "plugin exploded"
    end

    test "first plugin crash stops pipeline" do
      test_pid = self()

      crashing = fn _req -> raise "boom" end

      second = fn req ->
        send(test_pid, :second_called)
        req
      end

      {:error, _} =
        Generation.generate_text(
          "openai:gpt-4o-mini",
          "Hello",
          req_plugins: [crashing, second],
          req_http_options: [plug: {Req.Test, __MODULE__}]
        )

      refute_receive :second_called
    end
  end

  describe "generate_object/4 with :req_plugins" do
    test "plugins applied to object generation requests" do
      test_pid = self()

      Req.Test.stub(ReqLLM.ReqPluginsTest.ObjectPlugin, fn conn ->
        send(test_pid, {:headers, conn.req_headers})
        Req.Test.json(conn, @object_response)
      end)

      plugin = fn req ->
        Req.Request.put_header(req, "x-object-plugin", "active")
      end

      schema = [
        name: [type: :string],
        age: [type: :integer]
      ]

      {:ok, response} =
        Generation.generate_object(
          "openai:gpt-4o-mini",
          "Generate a person",
          schema,
          req_plugins: [plugin],
          req_http_options: [plug: {Req.Test, ReqLLM.ReqPluginsTest.ObjectPlugin}]
        )

      assert %Response{} = response
      assert_receive {:headers, headers}
      assert List.keyfind(headers, "x-object-plugin", 0) == {"x-object-plugin", "active"}
    end
  end

  describe "embed/3 with :req_plugins" do
    test "plugins applied to single text embedding" do
      test_pid = self()

      Req.Test.stub(ReqLLM.ReqPluginsTest.EmbedPlugin, fn conn ->
        send(test_pid, {:headers, conn.req_headers})
        Req.Test.json(conn, @embedding_response)
      end)

      plugin = fn req ->
        Req.Request.put_header(req, "x-embed-plugin", "yes")
      end

      {:ok, embedding} =
        Embedding.embed(
          "openai:text-embedding-3-small",
          "Hello world",
          req_plugins: [plugin],
          req_http_options: [plug: {Req.Test, ReqLLM.ReqPluginsTest.EmbedPlugin}]
        )

      assert is_list(embedding)
      assert_receive {:headers, headers}
      assert List.keyfind(headers, "x-embed-plugin", 0) == {"x-embed-plugin", "yes"}
    end

    test "plugins applied to batch embedding" do
      test_pid = self()

      batch_response = %{
        "object" => "list",
        "data" => [
          %{"object" => "embedding", "index" => 0, "embedding" => [0.1, 0.2]},
          %{"object" => "embedding", "index" => 1, "embedding" => [0.3, 0.4]}
        ],
        "model" => "text-embedding-3-small",
        "usage" => %{"prompt_tokens" => 10, "total_tokens" => 10}
      }

      Req.Test.stub(ReqLLM.ReqPluginsTest.BatchEmbedPlugin, fn conn ->
        send(test_pid, {:headers, conn.req_headers})
        Req.Test.json(conn, batch_response)
      end)

      plugin = fn req ->
        Req.Request.put_header(req, "x-batch-embed", "true")
      end

      {:ok, embeddings} =
        Embedding.embed(
          "openai:text-embedding-3-small",
          ["Hello", "World"],
          req_plugins: [plugin],
          req_http_options: [plug: {Req.Test, ReqLLM.ReqPluginsTest.BatchEmbedPlugin}]
        )

      assert length(embeddings) == 2
      assert_receive {:headers, headers}
      assert List.keyfind(headers, "x-batch-embed", 0) == {"x-batch-embed", "true"}
    end
  end

  describe "apply_req_plugins/2 unit" do
    test "returns {:ok, request} when no plugins" do
      req = Req.new(url: "https://example.com")
      assert {:ok, ^req} = ReqLLM.Provider.Defaults.apply_req_plugins(req, [])
    end

    test "returns {:ok, request} with empty plugin list" do
      req = Req.new(url: "https://example.com")
      assert {:ok, ^req} = ReqLLM.Provider.Defaults.apply_req_plugins(req, req_plugins: [])
    end

    test "applies plugin transformation" do
      req = Req.new(url: "https://example.com")

      plugin = fn r -> Req.Request.put_header(r, "x-test", "value") end

      assert {:ok, result} =
               ReqLLM.Provider.Defaults.apply_req_plugins(req, req_plugins: [plugin])

      assert Req.Request.get_header(result, "x-test") == ["value"]
    end

    test "returns error on plugin crash" do
      req = Req.new(url: "https://example.com")
      plugin = fn _r -> raise "kaboom" end

      assert {:error, error} =
               ReqLLM.Provider.Defaults.apply_req_plugins(req, req_plugins: [plugin])

      assert Exception.message(error) =~ "kaboom"
    end
  end
end
