// This function gets your whole document as its `body` and formats
// it as an article in the style of the IEEE.
//

#let green = rgb("#8bab82")
#let orange = rgb("#dd8c6e")

#let report(
  // The paper's title.
  title: "Paper Title",

  // An array of authors. For each author you can specify a name,
  // department, organization, location, and email. Everything but
  // but the name is optional.
  authors: (),

  // The paper's abstract. Can be omitted if you don't have one.
  abstract: none,

  // A list of index terms to display after the abstract.
  index-terms: (),

  // The article's paper size. Also affects the margins.
  paper-size: "a4",

  // The path to a bibliography file if you want to cite some external
  // works.
  bibliography-file: none,

  // The paper's content.
  body
) = {
  // Set document metadata.
  set document(title: title, author: authors.map(author => author.name))

  // Set the body font.
  set text(font: "New Computer Modern", size: 12pt)

  // Configure the page.
  set page(
    paper: paper-size,
    // The margins depend on the paper size.
    margin: (
        x: 40mm,
        top: 30mm,
        bottom: 30mm,
    ),
    numbering: "1 / 1",
  )

  // Configure equation numbering and spacing.
  set math.equation(numbering: "(1)", supplement: "")
  show math.equation: set block(spacing: 0.65em)

  // Configure lists.
  set enum(indent: 10pt, body-indent: 9pt)
  set list(indent: 10pt, body-indent: 9pt)

  // Configure headings.
  set heading(numbering: "1.1")
  show heading: it => locate(loc => {
    // Find out the final number of the heading counter.
    let levels = counter(heading).at(loc)
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    if it.level == 1 [
      #set text(16pt, weight: 400)
      #set align(left)
      #set par(first-line-indent: 0pt)
      #v(24pt, weak: true)
      #if it.numbering != none {
        numbering("1.", deepest)
        h(7pt, weak: true)
      }
      #it.body
      #v(12pt, weak: true)
    ] else if it.level == 2 [
      // Second-level headings are run-ins.
      #set text(13pt, weight: 400)
      #set par(first-line-indent: 0pt)
      // #set text(style: "italic")
      #v(12pt, weak: true)
      #if it.numbering != none {
        numbering("1.1", levels.at(0), levels.at(1))
        h(7pt, weak: true)
      }
      #it.body
      #v(10pt, weak: true)
    ] else [
      *#(it.body)*.
    ]
  })

  show raw: set text(size: 8pt)

  show figure: it => block(width: 100%)[#align(center)[
    #it.body
    #set align(left)
    #set text(size: 8pt)
    #set par(hanging-indent: 12pt)
    #[
      #set text(weight: "bold")
      #it.supplement
      #it.counter.display(it.numbering):
    ]
    _#it.caption _
  ]]

  // Display the paper's title.
  v(3pt, weak: true)
  align(center, text(24pt, title))
  v(9mm, weak: true)

  // Display the authors list.
  for i in range(calc.ceil(authors.len() / 3)) {
    let end = calc.min((i + 1) * 3, authors.len())
    let is-last = authors.len() == end
    let slice = authors.slice(i * 3, end)
    grid(
      columns: slice.len() * (1fr,),
      gutter: 12pt,
      ..slice.map(author => align(center, {
        text(12pt, author.name)
        if "department" in author [
          \ #emph(author.department)
        ]
        if "organization" in author [
          \ #emph(author.organization)
        ]
        if "location" in author [
          \ #author.location
        ]
        if "identifier" in author [
          \ #author.identifier
        ]
        if "email" in author [
          \ #link("mailto:" + author.email)
        ]
      }))
    )

    if not is-last {
      v(16pt, weak: true)
    }
  }
  v(45pt, weak: true)

  // Start two column mode and configure paragraph properties.
  // show: columns.with(2, gutter: 12pt)
  set par(justify: true, first-line-indent: 1em)
  show par: set block(spacing: 0.65em)

  // Display abstract and index terms.
  if abstract != none [
    #set text(weight: 700)
    #h(1em) _Abstract_---#abstract

    #if index-terms != () [
      #h(1em)_Index terms_---#index-terms.join(", ")
    ]
    #v(2pt)
  ]

  // Display the paper's contents.
  body

  // Display bibliography.
  if bibliography-file != none {
    show bibliography: set text(8pt)
    bibliography(bibliography-file, title: text(15pt)[References], style: "ieee")
  }
}

#let note(body) = {
  set text(green)
  [#body]
}

