from definers.cli.cli_request import CliRequest


class CliRequestCoercer:
    @staticmethod
    def coerce_cli_request(source: object | CliRequest) -> CliRequest:
        if isinstance(source, CliRequest):
            return source
        return CliRequest(
            command=getattr(source, "command", None),
            project=getattr(source, "project", "chat"),
            install_target=getattr(source, "install_target", ""),
            install_kind=getattr(source, "install_kind", "group"),
            install_list=getattr(source, "install_list", False),
            audio=getattr(source, "audio", ""),
            width=getattr(source, "width", 0),
            height=getattr(source, "height", 0),
            fps=getattr(source, "fps", 0),
            background=getattr(source, "background", ""),
            lyrics=getattr(source, "lyrics", ""),
            position=getattr(source, "position", "bottom"),
            max_dim=getattr(source, "max_dim", 640),
            font_size=getattr(source, "font_size", 70),
            text_color=getattr(source, "text_color", "white"),
            stroke_color=getattr(source, "stroke_color", "black"),
            stroke_width=getattr(source, "stroke_width", 2),
            fade=getattr(source, "fade", 0.5),
        )
