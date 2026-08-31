def test_claude_code_registered():
    import evaluate
    from orchestrators.claude_code import ClaudeCodeOrchestrator

    assert evaluate.ORCHESTRATOR_MAP["claude_code"] is ClaudeCodeOrchestrator
