// Book structure and chapter file mapping
const book = {
    introduction: {
        title: 'Introduction',
        file: 'introduction_draft.md',
    },
    chapters: [
        { title: 'Chapter 1: What Are LLMs and How Do They Work?', file: 'chapter1_draft.md' },
        { title: 'Chapter 2: The LLM Playground: Use Cases and Possibilities', file: 'chapter2_draft.md' },
        { title: 'Chapter 3: The Problem with Statelessness: Introducing Agents', file: 'chapter3_draft.md' },
        { title: 'Chapter 4: Building Your First Agent in Python', file: 'chapter4_draft.md' },
        { title: 'Chapter 5: Thinking Together: Multi-Agent Systems', file: 'chapter5_draft.md' },
        { title: 'Chapter 6: Remembering the Past: Context and Memory', file: 'chapter6_draft.md' },
        { title: 'Chapter 7: Grounding LLMs in Reality: RAG and CAG', file: 'chapter7_draft.md' },
        { title: 'Chapter 8: Making LLMs Your Own: Fine-Tuning Explained', file: 'chapter8_draft.md' },
        { title: 'Chapter 9: Choosing Your Tools: LLM Frameworks Deep Dive', file: 'chapter9_draft.md' },
        { title: 'Chapter 10: The Evolving Landscape and the Future', file: 'chapter10_draft.md' },
    ],
    conclusion: {
        title: 'Conclusion',
        file: 'conclusion_draft.md',
    }
};

const tocList = document.getElementById('toc-list');
const chapterContent = document.getElementById('chapter-content');

// Utility to fetch and render markdown (basic)
async function fetchMarkdown(file) {
    try {
        const resp = await fetch(file);
        if (!resp.ok) throw new Error('File not found');
        const text = await resp.text();
        return marked.parse(text); // Use marked.js for Markdown parsing
    } catch (err) {
        return `<p style='color:red'>Unable to load content.</p>`;
    }
}

// Build Table of Contents with research progress
async function buildTOC() {
    tocList.innerHTML = '';
    // Load todo.md and parse research completion
    let todoText = '';
    try {
        const resp = await fetch('todo.md');
        if (resp.ok) todoText = await resp.text();
    } catch {}
    // Helper to check if all research for a chapter is done
    function isChapterDone(chapterNum) {
        const regex = new RegExp(`# Chapter ${chapterNum} Research:[\s\S]*?(?=#|$)`, 'g');
        const section = (todoText.match(regex) || [''])[0];
        const checks = (section.match(/\[([ Xx])\]/g) || []);
        return checks.length > 0 && checks.every(c => c.includes('X'));
    }
    // Introduction
    const introLi = document.createElement('li');
    introLi.innerHTML = `<a href="#" data-type="intro">${book.introduction.title}</a>`;
    tocList.appendChild(introLi);
    // Chapters
    book.chapters.forEach((ch, idx) => {
        const li = document.createElement('li');
        const done = isChapterDone(idx + 1);
        li.innerHTML = `<a href="#" data-type="chapter" data-idx="${idx}" class="${done ? 'done' : ''}">${ch.title}${done ? ' <span class=\'check\' title=\'All research tasks done\'>&#10003;</span>' : ''}</a>`;
        tocList.appendChild(li);
    });
    // Conclusion
    const conclLi = document.createElement('li');
    conclLi.innerHTML = `<a href="#" data-type="conclusion">${book.conclusion.title}</a>`;
    tocList.appendChild(conclLi);
}



// Render content
async function renderContent(type, idx = null) {
    let file, tocIdx;
    if (type === 'intro') {
        file = book.introduction.file;
        tocIdx = 0;
    } else if (type === 'chapter') {
        file = book.chapters[idx].file;
        tocIdx = idx + 1;
    } else if (type === 'conclusion') {
        file = book.conclusion.file;
        tocIdx = book.chapters.length + 1;
    }
    // Highlight TOC
    Array.from(tocList.children).forEach((li, i) => {
        li.classList.toggle('active', i === tocIdx);
    });
    // Load and render markdown
    chapterContent.innerHTML = '<p>Loading...</p>';
    chapterContent.innerHTML = await fetchMarkdown(file);
}

// Initial load
async function initBook() {
    await buildTOC();
    renderContent('intro');
    // TOC click events
    tocList.addEventListener('click', (e) => {
        if (e.target.tagName === 'A') {
            const type = e.target.dataset.type;
            const idx = e.target.dataset.idx ? parseInt(e.target.dataset.idx) : null;
            renderContent(type, idx);
            e.preventDefault();
        }
    });
}

// Load marked.js for Markdown rendering
(function loadMarked() {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    script.onload = initBook; // Only call initBook after marked is loaded
    document.body.appendChild(script);
})();
