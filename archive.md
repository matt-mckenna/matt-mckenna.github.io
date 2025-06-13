---
layout: page
title: Archive
permalink: /archive
---

# Archived Posts

{% for post in site.archive %}
* [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %d, %Y" }}
{% endfor %}
