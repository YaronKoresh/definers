class ChatRequestAuditLogger:
    @staticmethod
    def log_request(log, build_audit_message, context) -> None:
        log("Chat", build_audit_message(context))

    @staticmethod
    def log_response(log, response) -> None:
        log("Chatbot response", response)