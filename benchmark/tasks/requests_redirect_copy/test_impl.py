import unittest
from impl import PreparedRequest, Response, resolve_redirects


class TestResolveRedirects(unittest.TestCase):

    def _post(self, url="http://example.com/api"):
        return PreparedRequest(method="POST", url=url, body=b"data=1")

    def test_single_303_switches_to_get(self):
        """After a 303, the redirect request must use GET."""
        req = self._post()
        r303 = Response(303, headers={"Location": "http://example.com/result"})
        history = resolve_redirects(Response(200), req, [r303])
        self.assertEqual(history[0].method, "GET")
        self.assertIsNone(history[0].body)

    def test_original_request_not_mutated(self):
        """The original PreparedRequest must not be modified."""
        req = self._post()
        r303 = Response(303, headers={"Location": "http://example.com/ok"})
        resolve_redirects(Response(200), req, [r303])
        self.assertEqual(req.method, "POST", "Original request method must stay POST")
        self.assertEqual(req.body, b"data=1", "Original request body must stay intact")

    def test_301_preserves_method(self):
        """301 does not change the method."""
        req = PreparedRequest(method="GET", url="http://example.com/old")
        r301 = Response(301, headers={"Location": "http://example.com/new"})
        history = resolve_redirects(Response(200), req, [r301])
        self.assertEqual(history[0].method, "GET")

    def test_two_redirects_independent(self):
        """Each redirect must produce an independent copy; mutations must not bleed."""
        req = self._post()
        r303 = Response(303, headers={"Location": "http://example.com/step2"})
        r301 = Response(301, headers={"Location": "http://example.com/final"})
        history = resolve_redirects(Response(200), req, [r303, r301])
        # First hop: 303 → GET
        self.assertEqual(history[0].method, "GET")
        # Second hop: 301 after GET should still be GET (not bleed from earlier POST)
        self.assertEqual(history[1].method, "GET")
        # Each hop must have a distinct object (not aliases)
        self.assertIsNot(history[0], history[1])


if __name__ == "__main__":
    unittest.main()
