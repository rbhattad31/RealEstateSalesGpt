import streamlit as st
import re


def add_newlines_around_tag(text, tag="<tag>"):
    pattern = rf'({re.escape(tag)})'
    replaced_text = re.sub(pattern, r'\n\1\n', text)
    return replaced_text


col1, mid, col2 = st.columns([20, 1, 20])
with col2:
    pass
    # st.image("https://t4.ftcdn.net/jpg/02/81/89/73/360_F_281897358_3rj9ZBSZHo5s0L1ug7uuIHadSxh9Cc75.jpg", width=100)
with col1:
    html_string = '<img src="https://t4.ftcdn.net/jpg/02/81/89/73/360_F_281897358_3rj9ZBSZHo5s0L1ug7uuIHadSxh9Cc75.jpg" width="440">'
    html_string = add_newlines_around_tag(html_string)
    st.chat_message('user').markdown(html_string, unsafe_allow_html=True)
#     #st.write("""Amenities : Balcony, BP, BB, AN, Built-in Wardrobes, Central A/C & Heating, Covered Parking, LF, Pets Allowed, Shared Pool, Security, MT, IC, BT, ML, PK, Walk-in Closet
# Size : 582
# Bedrooms : 0
# Bathrooms : 1
# Name : Rahul Bhatadttad
# Parking : 1m,jh,.k
# Furnished : No
# html_string = "<h3>this is an html string</h3>"
#
#
# st.markdown(html_string, unsafe_allow_html=True)
# <img src-https://t4.ftcdn.net/jpg/02/81/89/73/360_F_281897358_3rj9ZBSZHo5s0L1ug7uuIHadSxh9Cc75.jpg>""")
