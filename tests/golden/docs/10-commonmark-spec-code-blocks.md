---
title: CommonMark Spec
author: John MacFarlane
version: '0.31.2'
date: '2024-01-28'
license: '[CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)'
...

## Indented code blocks

An [indented code block](@) is composed of one or more
[indented chunks] separated by blank lines.
An [indented chunk](@) is a sequence of non-blank lines,
each preceded by four or more spaces of indentation. The contents of the code
block are the literal contents of the lines, including trailing
[line endings], minus four spaces of indentation.
An indented code block has no [info string].

An indented code block cannot interrupt a paragraph, so there must be
a blank line between a paragraph and a following indented code block.
(A blank line is not needed, however, between a code block and a following
paragraph.)

```````````````````````````````` example
    a simple
      indented code block
.
<pre><code>a simple
  indented code block
</code></pre>
````````````````````````````````


If there is any ambiguity between an interpretation of indentation
as a code block and as indicating that material belongs to a [list
item][list items], the list item interpretation takes precedence:

```````````````````````````````` example
  - foo

    bar
.
<ul>
<li>
<p>foo</p>
<p>bar</p>
</li>
</ul>
````````````````````````````````


```````````````````````````````` example
1.  foo

    - bar
.
<ol>
<li>
<p>foo</p>
<ul>
<li>bar</li>
</ul>
</li>
</ol>
````````````````````````````````



The contents of a code block are literal text, and do not get parsed
as Markdown:

```````````````````````````````` example
    <a/>
    *hi*

    - one
.
<pre><code>&lt;a/&gt;
*hi*

- one
</code></pre>
````````````````````````````````


Here we have three chunks separated by blank lines:

```````````````````````````````` example
    chunk1

    chunk2
  
 
 
    chunk3
.
<pre><code>chunk1

chunk2



chunk3
</code></pre>
````````````````````````````````


Any initial spaces or tabs beyond four spaces of indentation will be included in
the content, even in interior blank lines:

```````````````````````````````` example
    chunk1
      
      chunk2
.
<pre><code>chunk1
  
  chunk2
</code></pre>
````````````````````````````````


An indented code block cannot interrupt a paragraph.  (This
allows hanging indents and the like.)

```````````````````````````````` example
Foo
    bar

.
<p>Foo
bar</p>
````````````````````````````````


However, any non-blank line with fewer than four spaces of indentation ends
the code block immediately.  So a paragraph may occur immediately
after indented code:

```````````````````````````````` example
    foo
bar
.
<pre><code>foo
</code></pre>
<p>bar</p>
````````````````````````````````


And indented code can occur immediately before and after other kinds of
blocks:

```````````````````````````````` example
# Heading
    foo
Heading
------
    foo
----
.
<h1>Heading</h1>
<pre><code>foo
</code></pre>
<h2>Heading</h2>
<pre><code>foo
</code></pre>
<hr />
````````````````````````````````


The first line can be preceded by more than four spaces of indentation:

```````````````````````````````` example
        foo
    bar
.
<pre><code>    foo
bar
</code></pre>
````````````````````````````````


Blank lines preceding or following an indented code block
are not included in it:

```````````````````````````````` example

    
    foo
    

.
<pre><code>foo
</code></pre>
````````````````````````````````


Trailing spaces or tabs are included in the code block's content:

```````````````````````````````` example
    foo  
.
<pre><code>foo  
</code></pre>
````````````````````````````````



## Fenced code blocks

A [code fence](@) is a sequence
of at least three consecutive backtick characters (`` ` ``) or
tildes (`~`).  (Tildes and backticks cannot be mixed.)
A [fenced code block](@)
begins with a code fence, preceded by up to three spaces of indentation.

The line with the opening code fence may optionally contain some text
following the code fence; this is trimmed of leading and trailing
spaces or tabs and called the [info string](@). If the [info string] comes
after a backtick fence, it must not contain any backtick
characters.  (The reason for this restriction is that otherwise
some inline code would be incorrectly interpreted as the
beginning of a fenced code block.)

The content of the code block consists of all subsequent lines, until
a closing [code fence] of the same type as the code block
began with (backticks or tildes), and with at least as many backticks
or tildes as the opening code fence.  If the leading code fence is
preceded by N spaces of indentation, then up to N spaces of indentation are
removed from each line of the content (if present).  (If a content line is not
indented, it is preserved unchanged.  If it is indented N spaces or less, all
of the indentation is removed.)

The closing code fence may be preceded by up to three spaces of indentation, and
may be followed only by spaces or tabs, which are ignored.  If the end of the
containing block (or document) is reached and no closing code fence
has been found, the code block contains all of the lines after the
opening code fence until the end of the containing block (or
document).  (An alternative spec would require backtracking in the
event that a closing code fence is not found.  But this makes parsing
much less efficient, and there seems to be no real downside to the
behavior described here.)

A fenced code block may interrupt a paragraph, and does not require
a blank line either before or after.

The content of a code fence is treated as literal text, not parsed
as inlines.  The first word of the [info string] is typically used to
specify the language of the code sample, and rendered in the `class`
attribute of the `code` tag.  However, this spec does not mandate any
particular treatment of the [info string].

Here is a simple example with backticks:

```````````````````````````````` example
```
<
 >
```
.
<pre><code>&lt;
 &gt;
</code></pre>
````````````````````````````````


With tildes:

```````````````````````````````` example
~~~
<
 >
~~~
.
<pre><code>&lt;
 &gt;
</code></pre>
````````````````````````````````

Fewer than three backticks is not enough:

```````````````````````````````` example
``
foo
``
.
<p><code>foo</code></p>
````````````````````````````````

The closing code fence must use the same character as the opening
fence:

```````````````````````````````` example
```
aaa
~~~
```
.
<pre><code>aaa
~~~
</code></pre>
````````````````````````````````


```````````````````````````````` example
~~~
aaa
```
~~~
.
<pre><code>aaa
```
</code></pre>
````````````````````````````````


The closing code fence must be at least as long as the opening fence:

```````````````````````````````` example
````
aaa
```
``````
.
<pre><code>aaa
```
</code></pre>
````````````````````````````````


```````````````````````````````` example
~~~~
aaa
~~~
~~~~
.
<pre><code>aaa
~~~
</code></pre>
````````````````````````````````


Unclosed code blocks are closed by the end of the document
(or the enclosing [block quote][block quotes] or [list item][list items]):

```````````````````````````````` example
```
.
<pre><code></code></pre>
````````````````````````````````


```````````````````````````````` example
`````

```
aaa
.
<pre><code>
```
aaa
</code></pre>
````````````````````````````````


```````````````````````````````` example
> ```
> aaa

bbb
.
<blockquote>
<pre><code>aaa
</code></pre>
</blockquote>
<p>bbb</p>
````````````````````````````````


A code block can have all empty lines as its content:

```````````````````````````````` example
```

  
```
.
<pre><code>
  
</code></pre>
````````````````````````````````


A code block can be empty:

```````````````````````````````` example
```
```
.
<pre><code></code></pre>
````````````````````````````````


Fences can be indented.  If the opening fence is indented,
content lines will have equivalent opening indentation removed,
if present:

```````````````````````````````` example
 ```
 aaa
aaa
```
.
<pre><code>aaa
aaa
</code></pre>
````````````````````````````````


```````````````````````````````` example
  ```
aaa
  aaa
aaa
  ```
.
<pre><code>aaa
aaa
aaa
</code></pre>
````````````````````````````````


```````````````````````````````` example
   ```
   aaa
    aaa
  aaa
   ```
.
<pre><code>aaa
 aaa
aaa
</code></pre>
````````````````````````````````


Four spaces of indentation is too many:

```````````````````````````````` example
    ```
    aaa
    ```
.
<pre><code>```
aaa
```
</code></pre>
````````````````````````````````


Closing fences may be preceded by up to three spaces of indentation, and their
indentation need not match that of the opening fence:

```````````````````````````````` example
```
aaa
  ```
.
<pre><code>aaa
</code></pre>
````````````````````````````````


```````````````````````````````` example
   ```
aaa
  ```
.
<pre><code>aaa
</code></pre>
````````````````````````````````


This is not a closing fence, because it is indented 4 spaces:

```````````````````````````````` example
```
aaa
    ```
.
<pre><code>aaa
    ```
</code></pre>
````````````````````````````````



Code fences (opening and closing) cannot contain internal spaces or tabs:

```````````````````````````````` example
``` ```
aaa
.
<p><code> </code>
aaa</p>
````````````````````````````````


```````````````````````````````` example
~~~~~~
aaa
~~~ ~~
.
<pre><code>aaa
~~~ ~~
</code></pre>
````````````````````````````````


Fenced code blocks can interrupt paragraphs, and can be followed
directly by paragraphs, without a blank line between:

```````````````````````````````` example
foo
```
bar
```
baz
.
<p>foo</p>
<pre><code>bar
</code></pre>
<p>baz</p>
````````````````````````````````


Other blocks can also occur before and after fenced code blocks
without an intervening blank line:

```````````````````````````````` example
foo
---
~~~
bar
~~~
# baz
.
<h2>foo</h2>
<pre><code>bar
</code></pre>
<h1>baz</h1>
````````````````````````````````


An [info string] can be provided after the opening code fence.
Although this spec doesn't mandate any particular treatment of
the info string, the first word is typically used to specify
the language of the code block. In HTML output, the language is
normally indicated by adding a class to the `code` element consisting
of `language-` followed by the language name.

```````````````````````````````` example
```ruby
def foo(x)
  return 3
end
```
.
<pre><code class="language-ruby">def foo(x)
  return 3
end
</code></pre>
````````````````````````````````


```````````````````````````````` example
~~~~    ruby startline=3 $%@#$
def foo(x)
  return 3
end
~~~~~~~
.
<pre><code class="language-ruby">def foo(x)
  return 3
end
</code></pre>
````````````````````````````````


```````````````````````````````` example
````;
````
.
<pre><code class="language-;"></code></pre>
````````````````````````````````


[Info strings] for backtick code blocks cannot contain backticks:

```````````````````````````````` example
``` aa ```
foo
.
<p><code>aa</code>
foo</p>
````````````````````````````````


[Info strings] for tilde code blocks can contain backticks and tildes:

```````````````````````````````` example
~~~ aa ``` ~~~
foo
~~~
.
<pre><code class="language-aa">foo
</code></pre>
````````````````````````````````


Closing code fences cannot have [info strings]:

```````````````````````````````` example
```
``` aaa
```
.
<pre><code>``` aaa
</code></pre>
````````````````````````````````


