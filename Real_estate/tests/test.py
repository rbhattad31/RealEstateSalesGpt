import streamlit as st
import re


def add_newlines_around_tag(text, tag="<tag>"):
    pattern = rf'({re.escape(tag)})'
    replaced_text = re.sub(pattern, r'\n\1\n', text)
    return replaced_text

html_string = """Here is the image of the apartment at 1 Residences, Al Kifaf, Dubai that you requested: 
<img src="https://s3-ap-southeast-1.amazonaws.com/mycrm-pro-accounts-v2/property/full/805/72bc072c-a47b-11ed-bc98-d2b32331e5e4.jpeg" width="550"> Is there anything else I can help you 
with? <END_OF_TURN>"""
pattern = r'<([^>]+)>'

# Find the match using the pattern
match = re.search(pattern, html_string)

if match:
    img_src = match.group(1)
    print("Image URL:", img_src)

else:
    pass
html_string = add_newlines_around_tag(html_string)
st.chat_message('user').markdown(html_string, unsafe_allow_html=True)

