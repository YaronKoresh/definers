class RegressionPredictor:
    @staticmethod
    def catch(error: Exception) -> None:
        try:
            from definers.system import catch as runtime_catch

            runtime_catch(error)
        except Exception:
            return None

    @classmethod
    def sanitize_path(cls, model_path: str):
        from definers.system import secure_path

        try:
            return secure_path(model_path)
        except Exception as error:
            cls.catch(error)
            return None

    @classmethod
    def predict(cls, X_new, model_path: str, *, factory):
        import torch

        sanitized_path = cls.sanitize_path(model_path)
        if sanitized_path is None:
            return None
        try:
            input_dim = X_new.shape[1]
            model_torch = factory(input_dim)
            model_torch.load_state_dict(
                torch.load(sanitized_path, map_location="cpu")
            )
            model_torch.eval()
            target_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            if hasattr(model_torch, "to"):
                model_torch.to(target_device)
            X_new_torch = torch.tensor(
                X_new,
                dtype=torch.float32,
                device=target_device,
            )
            with torch.no_grad():
                predictions_torch = model_torch(X_new_torch).reshape(-1)
            return predictions_torch.cpu().numpy().reshape(-1)
        except Exception as error:
            cls.catch(error)
            return None