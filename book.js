// Book structure and chapter file mapping
const book = {
  introduction: {
    title: "Introduction",
    file: "docs/introduction_draft.md",
  },
  chapters: [
    {
      title: "Chapter 1: What Are LLMs and How Do They Work?",
      file: "docs/chapter1_draft.md",
      researchFile: "docs/chapter1_research_notes.md",
    },
    {
      title: "Chapter 2: The LLM Playground: Use Cases and Possibilities",
      file: "docs/chapter2_draft.md",
      researchFile: "docs/chapter2_research_notes.md",
    },
    {
      title: "Chapter 3: The Problem with Statelessness: Introducing Agents",
      file: "docs/chapter3_draft.md",
      researchFile: "docs/chapter3_research_notes.md",
    },
    {
      title: "Chapter 4: Building Your First Agent in Python",
      file: "docs/chapter4_draft.md",
      researchFile: "docs/chapter4_research_notes.md",
    },
    {
      title: "Chapter 5: Thinking Together: Multi-Agent Systems",
      file: "docs/chapter5_draft.md",
      researchFile: "docs/chapter5_research_notes.md",
    },
    {
      title: "Chapter 6: Remembering the Past: Context and Memory",
      file: "docs/chapter6_draft.md",
      researchFile: "docs/chapter6_research_notes.md",
    },
    {
      title: "Chapter 7: Grounding LLMs in Reality: RAG and CAG",
      file: "docs/chapter7_draft.md",
      researchFile: "docs/chapter7_research_notes.md",
    },
    {
      title: "Chapter 8: Making LLMs Your Own: Fine-Tuning Explained",
      file: "docs/chapter8_draft.md",
      researchFile: "docs/chapter8_research_notes.md",
    },
    {
      title: "Chapter 9: Choosing Your Tools: LLM Frameworks Deep Dive",
      file: "docs/chapter9_draft.md",
      researchFile: "docs/chapter9_research_notes.md",
    },
    {
      title: "Chapter 10: The Evolving Landscape and the Future",
      file: "docs/chapter10_draft.md",
      researchFile: null,
    },
  ],
  conclusion: {
    title: "Conclusion",
    file: "docs/conclusion_draft.md",
  },
};

// DOM elements
const tocList = document.getElementById("toc-list");
const chapterMd = document.getElementById("chapter-md");
const researchMd = document.getElementById("research-md");
const researchNotesContainer = document.getElementById(
  "research-notes-container"
);
const showResearchNotesCheckbox = document.getElementById(
  "show-research-notes"
);
const currentChapterTitle = document.getElementById("current-chapter-title");
const prevChapterBtn = document.getElementById("prev-chapter");
const nextChapterBtn = document.getElementById("next-chapter");
const prevChapterBtnBottom = document.getElementById("prev-chapter-bottom");
const nextChapterBtnBottom = document.getElementById("next-chapter-bottom");

// Settings elements
const settingsToggle = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings-panel");
const fontSizeSmaller = document.getElementById("font-size-smaller");
const fontSizeLarger = document.getElementById("font-size-larger");
const fontSizeDisplay = document.getElementById("font-size-display");
const themeLight = document.getElementById("theme-light");
const themeSepia = document.getElementById("theme-sepia");
const themeDark = document.getElementById("theme-dark");

// Current position tracker
let currentPosition = {
  type: "intro",
  idx: null,
};

// Build Table of Contents with research progress
async function buildTOC() {
  tocList.innerHTML = "";
  // Load todo.md and parse research completion
  let todoText = "";
  try {
    const resp = await fetch("docs/todo.md");
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
function renderContent(type, idx = null) {
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

  // Load markdown content
  loadMarkdownContent(file, researchFile);

  // Update navigation buttons
  updateNavigationButtons();
}

// Load markdown content using zero-md
function loadMarkdownContent(file, researchFile) {
  console.log(`Loading markdown file: ${file}`);

  // Clear previous content
  if (chapterMd._shadowRoot) {
    chapterMd.src = "";
  }

  // Set the source for the chapter markdown
  setTimeout(() => {
    chapterMd.setAttribute("src", file);
  }, 50);

  // Update research notes if available
  if (researchFile && showResearchNotesCheckbox.checked) {
    console.log(`Loading research notes: ${researchFile}`);

    // Clear previous content
    if (researchMd._shadowRoot) {
      researchMd.src = "";
    }

    // Set the source for the research notes markdown
    setTimeout(() => {
      researchMd.setAttribute("src", researchFile);
    }, 50);

    researchNotesContainer.style.display = "block";
  } else {
    researchNotesContainer.style.display = "none";
  }
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

// Settings panel toggle
function toggleSettingsPanel() {
  settingsPanel.classList.toggle("active");
}

// Theme management
function setTheme(theme) {
  const bookApp = document.getElementById("book-app");
  bookApp.className = `theme-${theme}`;

  // Update active button
  themeLight.classList.toggle("active", theme === "light");
  themeSepia.classList.toggle("active", theme === "sepia");
  themeDark.classList.toggle("active", theme === "dark");

  // Save preference to localStorage
  localStorage.setItem("bookTheme", theme);
}

// Font size management
function adjustFontSize(direction) {
  const bookApp = document.getElementById("book-app");
  let currentSize = "medium";

  if (bookApp.classList.contains("font-small")) {
    currentSize = "small";
  } else if (bookApp.classList.contains("font-large")) {
    currentSize = "large";
  }

  // Calculate new size
  let newSize = currentSize;
  if (direction === "increase" && currentSize !== "large") {
    newSize = currentSize === "small" ? "medium" : "large";
  } else if (direction === "decrease" && currentSize !== "small") {
    newSize = currentSize === "large" ? "medium" : "small";
  }

  // Apply new size
  bookApp.classList.remove("font-small", "font-medium", "font-large");
  if (newSize !== "medium") {
    bookApp.classList.add(`font-${newSize}`);
  }

  // Update display
  fontSizeDisplay.textContent =
    newSize.charAt(0).toUpperCase() + newSize.slice(1);

  // Save preference to localStorage
  localStorage.setItem("bookFontSize", newSize);
}

// Load user preferences
function loadUserPreferences() {
  const theme = localStorage.getItem("bookTheme") || "light";
  const fontSize = localStorage.getItem("bookFontSize") || "medium";

  // Apply theme
  setTheme(theme);

  // Apply font size
  const bookApp = document.getElementById("book-app");
  bookApp.classList.remove("font-small", "font-medium", "font-large");
  if (fontSize !== "medium") {
    bookApp.classList.add(`font-${fontSize}`);
  }
  fontSizeDisplay.textContent =
    fontSize.charAt(0).toUpperCase() + fontSize.slice(1);
}

// Initial load function
function initBook() {
  // Build the table of contents
  buildTOC();

  // Load user preferences
  loadUserPreferences();

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

  // Settings events
  settingsToggle.addEventListener("click", toggleSettingsPanel);
  themeLight.addEventListener("click", () => setTheme("light"));
  themeSepia.addEventListener("click", () => setTheme("sepia"));
  themeDark.addEventListener("click", () => setTheme("dark"));
  fontSizeSmaller.addEventListener("click", () => adjustFontSize("decrease"));
  fontSizeLarger.addEventListener("click", () => adjustFontSize("increase"));

  // Close settings when clicking outside
  document.addEventListener("click", (e) => {
    if (
      settingsPanel.classList.contains("active") &&
      !settingsPanel.contains(e.target) &&
      e.target !== settingsToggle
    ) {
      settingsPanel.classList.remove("active");
    }
  });

  // Initial render
  renderContent("intro");
}

// Initialize when DOM is loaded
document.addEventListener("DOMContentLoaded", initBook);
