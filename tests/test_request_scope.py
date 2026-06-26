import unittest

from graph.request_scope import build_request_messages


class RequestScopeTests(unittest.TestCase):
    def test_build_request_messages_returns_single_fresh_user_turn(self) -> None:
        prompt = "같은 고객, 같은 질문은 항상 같은 fresh state로 실행되어야 합니다."

        self.assertEqual(build_request_messages(prompt), [("user", prompt)])


if __name__ == "__main__":
    unittest.main()
