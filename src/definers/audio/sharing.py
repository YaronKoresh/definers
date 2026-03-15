from urllib.parse import quote


def create_share_links(
    hf_username: str, space_name: str, file_path: str, text_description: str
) -> str:
    file_url = f"https://{hf_username}-{space_name}.hf.space/gradio_api/file={file_path}"
    encoded_text = quote(text_description)
    encoded_url = quote(file_url)
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}"
    facebook_link = (
        f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}"
    )
    reddit_link = (
        f"https://www.reddit.com/submit?url={encoded_url}&title={encoded_text}"
    )
    whatsapp_link = (
        f"https://api.whatsapp.com/send?text={encoded_text}%20{encoded_url}"
    )
    return (
        f"<div style='text-align:center; padding-top: 10px;'><p style='font-weight: bold;'>Share your creation!</p>"
        f"<a href='{twitter_link}' target='_blank' style='margin: 0 5px;'>X/Twitter</a> | "
        f"<a href='{facebook_link}' target='_blank' style='margin: 0 5px;'>Facebook</a> | "
        f"<a href='{reddit_link}' target='_blank' style='margin: 0 5px;'>Reddit</a> | "
        f"<a href='{whatsapp_link}' target='_blank' style='margin: 0 5px;'>WhatsApp</a></div>"
    )
