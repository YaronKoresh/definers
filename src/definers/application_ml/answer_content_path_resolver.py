class AnswerContentPathResolver:
    @staticmethod
    def content_paths(content: object) -> list[str]:
        if isinstance(content, dict):
            path = content.get("path")
            return [] if path is None else [str(path)]
        if isinstance(content, tuple):
            return [
                str(item["path"])
                for item in content
                if isinstance(item, dict) and item.get("path") is not None
            ]
        return []
