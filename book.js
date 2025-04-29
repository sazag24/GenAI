// Book structure and chapter file mapping
const book = {
  introduction: {
    title: "Introduction",
    file: "./docs/introduction_draft.md",
  },
  chapters: [
    {
      title: "Chapter 1: What Are LLMs and How Do They Work?",
      file: "./docs/chapter1_draft.md",
      researchFile: "./docs/chapter1_research_notes.md",
    },
    {
      title: "Chapter 2: The LLM Playground: Use Cases and Possibilities",
      file: "./docs/chapter2_draft.md",
      researchFile: "./docs/chapter2_research_notes.md",
    },
    {
      title: "Chapter 3: The Problem with Statelessness: Introducing Agents",
      file: "./docs/chapter3_draft.md",
      researchFile: "./docs/chapter3_research_notes.md",
    },
    {
      title: "Chapter 4: Building Your First Agent in Python",
      file: "./docs/chapter4_draft.md",
      researchFile: "./docs/chapter4_research_notes.md",
    },
    {
      title: "Chapter 5: Thinking Together: Multi-Agent Systems",
      file: "./docs/chapter5_draft.md",
      researchFile: "./docs/chapter5_research_notes.md",
    },
    {
      title: "Chapter 6: Remembering the Past: Context and Memory",
      file: "./docs/chapter6_draft.md",
      researchFile: "./docs/chapter6_research_notes.md",
    },
    {
      title: "Chapter 7: Grounding LLMs in Reality: RAG and CAG",
      file: "./docs/chapter7_draft.md",
      researchFile: "./docs/chapter7_research_notes.md",
    },
    {
      title: "Chapter 8: Making LLMs Your Own: Fine-Tuning Explained",
      file: "./docs/chapter8_draft.md",
      researchFile: "./docs/chapter8_research_notes.md",
    },
    {
      title: "Chapter 9: Choosing Your Tools: LLM Frameworks Deep Dive",
      file: "./docs/chapter9_draft.md",
      researchFile: "./docs/chapter9_research_notes.md",
    },
    {
      title: "Chapter 10: The Evolving Landscape and the Future",
      file: "./docs/chapter10_draft.md",
      researchFile: null,
    },
  ],
  conclusion: {
    title: "Conclusion",
    file: "./docs/conclusion_draft.md",
  },
};

// DOM elements
const tocList = document.getElementById("toc-list");
const chapterContent = document.getElementById("chapter-content");
const researchNotesContainer = document.getElementById(
  "research-notes-container"
);
const researchNotesContent = document.getElementById("research-notes-content");
const showResearchNotesCheckbox = document.getElementById(
  "show-research-notes"
);
const currentChapterTitle = document.getElementById("current-chapter-title");
const prevChapterBtn = document.getElementById("prev-chapter");
const nextChapterBtn = document.getElementById("next-chapter");
const prevChapterBtnBottom = document.getElementById("prev-chapter-bottom");
const nextChapterBtnBottom = document.getElementById("next-chapter-bottom");

// Current position tracker
let currentPosition = {
  type: "intro",
  idx: null,
};

// Utility to fetch and render markdown with better error handling
async function fetchMarkdown(file) {
  try {
    console.log("Fetching:", file);
    const resp = await fetch(file);
    if (!resp.ok) {
      console.error("Error fetching file:", file, resp.status);
      throw new Error(`File not found: ${resp.status}`);
    }
    const text = await resp.text();
    // Basic markdown to HTML conversion if marked.js isn't loaded
    if (typeof marked === "undefined") {
      console.warn("Marked.js not loaded, using basic conversion");
      return basicMarkdownToHtml(text);
    }
    return marked.parse(text);
  } catch (err) {
    console.error("Fetch error:", err);
    return `<p style='color:red'>Unable to load content: ${file}. ${err.message}</p>
            <p style='color:blue'>Please make sure files are in the correct location and the server has permission to access them.</p>`;
  }
}

// Basic markdown to HTML conversion as fallback
function basicMarkdownToHtml(markdown) {
  // Handle headings
  let html = markdown
    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
    // Handle bold and italic
    .replace(/\*\*(.*)\*\*/gm, '<strong>$1</strong>')
    .replace(/\*(.*)\*/gm, '<em>$1</em>')
    // Handle lists
    .replace(/^\*\s(.*$)/gm, '<li>$1</li>')
    // Handle paragraphs
    .replace(/^(?!<h|<li|<ul|<p)(.*$)/gm, '<p>$1</p>');
  
  // Wrap lists
  html = html.replace(/<li>(.*)<\/li>/gm, function(match) {
    return '<ul>' + match + '</ul>';
  });
  
  return html;
}

// Build Table of Contents with research progress
async function buildTOC() {
  tocList.innerHTML = "";
  // Load todo.md and parse research completion
  let todoText = "";
  try {
    const resp = await fetch("./docs/todo.md");
    if (resp.ok) todoText = await resp.text();
  } catch (e) {
    console.error("Failed to load todo.md", e);
  }

  // Helper to check if all research for a chapter is done
  function isChapterDone(chapterNum) {
    const regex = new RegExp(
      `# Chapter ${chapterNum} Research:([\\s\\S]*?)(?=#|$)`,
      "g"
    );
    const section = (todoText.match(regex) || [""])[0];
    const checks = section.match(/\[([ Xx])\]/g) || [];
    return checks.length > 0 && checks.every((c) => c.includes("X"));
  }

  // Introduction
  const introLi = document.createElement("li");
  introLi.innerHTML = `<a href="#" data-type="intro">${book.introduction.title}</a>`;
  tocList.appendChild(introLi);

  // Chapters
  book.chapters.forEach((ch, idx) => {
    const li = document.createElement("li");
    const done = isChapterDone(idx + 1);
    li.innerHTML = `<a href="#" data-type="chapter" data-idx="${idx}" class="${
      done ? "done" : ""
    }">${ch.title}${
      done
        ? " <span class='check' title='All research tasks done'>&#10003;</span>"
        : ""
    }</a>`;
    tocList.appendChild(li);
  });

  // Conclusion
  const conclLi = document.createElement("li");
  conclLi.innerHTML = `<a href="#" data-type="conclusion">${book.conclusion.title}</a>`;
  tocList.appendChild(conclLi);
}

// Render content and update UI
async function renderContent(type, idx = null) {
  currentPosition = { type, idx };

  let file, researchFile, title;

  // Set file paths and title based on content type
  if (type === "intro") {
    file = book.introduction.file;
    researchFile = null;
    title = book.introduction.title;
  } else if (type === "chapter") {
    file = book.chapters[idx].file;
    researchFile = book.chapters[idx].researchFile;
    title = book.chapters[idx].title;
  } else if (type === "conclusion") {
    file = book.conclusion.file;
    researchFile = null;
    title = book.conclusion.title;
  }

  // Update displayed title
  currentChapterTitle.textContent = title;

  // Update TOC highlight
  updateTOCHighlight();

  // Load and render content
  chapterContent.innerHTML = "<p>Loading...</p>";
  chapterContent.innerHTML = await fetchMarkdown(file);

  // Update research notes if available
  if (researchFile && showResearchNotesCheckbox.checked) {
    researchNotesContent.innerHTML = "<p>Loading research notes...</p>";
    researchNotesContent.innerHTML = await fetchMarkdown(researchFile);
    researchNotesContainer.style.display = "block";
  } else {
    researchNotesContainer.style.display = "none";
  }

  // Update navigation buttons
  updateNavigationButtons();
}

// Update TOC highlighting
function updateTOCHighlight() {
  // Get index to highlight in TOC
  let tocIdx;
  if (currentPosition.type === "intro") {
    tocIdx = 0;
  } else if (currentPosition.type === "chapter") {
    tocIdx = currentPosition.idx + 1;
  } else if (currentPosition.type === "conclusion") {
    tocIdx = book.chapters.length + 1;
  }

  // Apply active class
  Array.from(tocList.children).forEach((li, i) => {
    li.classList.toggle("active", i === tocIdx);
  });
}

// Update navigation button states
function updateNavigationButtons() {
  // Determine if we can navigate prev/next
  const canGoPrev =
    currentPosition.type === "chapter" || currentPosition.type === "conclusion";
  const canGoNext =
    currentPosition.type === "intro" ||
    (currentPosition.type === "chapter" &&
      currentPosition.idx < book.chapters.length - 1);

  // Set button states
  prevChapterBtn.disabled = !canGoPrev;
  nextChapterBtn.disabled = !canGoNext;
  prevChapterBtnBottom.disabled = !canGoPrev;
  nextChapterBtnBottom.disabled = !canGoNext;
}

// Navigate to previous chapter
function navigatePrev() {
  if (currentPosition.type === "chapter" && currentPosition.idx === 0) {
    // Go to intro
    renderContent("intro");
  } else if (currentPosition.type === "chapter") {
    // Go to previous chapter
    renderContent("chapter", currentPosition.idx - 1);
  } else if (currentPosition.type === "conclusion") {
    // Go to last chapter
    renderContent("chapter", book.chapters.length - 1);
  }
}

// Navigate to next chapter
function navigateNext() {
  if (currentPosition.type === "intro") {
    // Go to first chapter
    renderContent("chapter", 0);
  } else if (
    currentPosition.type === "chapter" &&
    currentPosition.idx < book.chapters.length - 1
  ) {
    // Go to next chapter
    renderContent("chapter", currentPosition.idx + 1);
  } else if (
    currentPosition.type === "chapter" &&
    currentPosition.idx === book.chapters.length - 1
  ) {
    // Go to conclusion
    renderContent("conclusion");
  }
}

// Initial load function
async function initBook() {
  // Build the table of contents
  await buildTOC();

  // Set up event listeners

  // TOC click events
  tocList.addEventListener("click", (e) => {
    if (e.target.tagName === "A") {
      const type = e.target.dataset.type;
      const idx = e.target.dataset.idx ? parseInt(e.target.dataset.idx) : null;
      renderContent(type, idx);
      e.preventDefault();
    }
  });

  // Navigation button clicks
  prevChapterBtn.addEventListener("click", navigatePrev);
  nextChapterBtn.addEventListener("click", navigateNext);
  prevChapterBtnBottom.addEventListener("click", navigatePrev);
  nextChapterBtnBottom.addEventListener("click", navigateNext);

  // Research notes toggle
  showResearchNotesCheckbox.addEventListener("change", () => {
    // Refresh content to show/hide research notes
    renderContent(currentPosition.type, currentPosition.idx);
  });

  // Initial render
  renderContent("intro");
}

// Load marked.js for Markdown rendering
(function loadMarked() {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    script.onload = initBook; // Only call initBook after marked is loaded
    document.body.appendChild(script);
})();
