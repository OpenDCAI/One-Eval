import React from "react";

// --- Simple Markdown Component ---

// Recursive Inline Parser
const parseInline = (text: string): React.ReactNode[] => {
    if (!text) return [];

    const elements: React.ReactNode[] = [];
    let i = 0;
    let buffer = "";

    const flushBuffer = () => {
        if (buffer) {
            elements.push(<span key={elements.length}>{buffer}</span>);
            buffer = "";
        }
    };

    while (i < text.length) {
        // 1. Code Block: `code`
        if (text[i] === '`') {
            const nextTick = text.indexOf('`', i + 1);
            if (nextTick !== -1) {
                flushBuffer();
                const codeContent = text.slice(i + 1, nextTick);
                elements.push(
                    <code key={elements.length} className="bg-slate-100 px-1.5 py-0.5 rounded text-pink-600 font-mono text-[11px] border border-slate-200 mx-0.5">
                        {codeContent}
                    </code>
                );
                i = nextTick + 1;
                continue;
            }
        }

        // 2. Bold: **text**
        if (text.startsWith("**", i)) {
            const nextBold = text.indexOf("**", i + 2);
            if (nextBold !== -1) {
                flushBuffer();
                const boldContent = text.slice(i + 2, nextBold);
                elements.push(
                    <strong key={elements.length} className="font-bold text-slate-900">
                        {parseInline(boldContent)}
                    </strong>
                );
                i = nextBold + 2;
                continue;
            }
        }

        // 3. Italic: *text* (Simple check: no space after first *, no space before last *)
        // Avoiding math expressions like 2 * 3.
        if (text[i] === '*') {
            const nextStar = text.indexOf('*', i + 1);
            if (nextStar !== -1) {
                // Heuristic: Check if it looks like markup
                // e.g. *italic* vs 2 * 3
                // We'll require it to NOT be surrounded by spaces if it's single word, 
                // or just simple matching for now.
                // Let's implement a safer check: 
                // Content must not start with space, must not end with space.
                const content = text.slice(i + 1, nextStar);
                if (content && !content.startsWith(' ') && !content.endsWith(' ')) {
                    flushBuffer();
                    elements.push(
                        <em key={elements.length} className="italic text-slate-700">
                            {parseInline(content)}
                        </em>
                    );
                    i = nextStar + 1;
                    continue;
                }
            }
        }

        buffer += text[i];
        i++;
    }
    flushBuffer();
    return elements;
};

export const SimpleMarkdown = ({ content }: { content: string }) => {
    if (!content) return null;

    // Split into lines to process block structures
    const lines = content.split('\n');
    const blocks: React.ReactNode[] = [];
    
    let currentListType: 'ul' | 'ol' | null = null;
    let currentListItems: React.ReactNode[] = [];

    const flushList = () => {
        if (!currentListType || currentListItems.length === 0) return;
        const Key = currentListType === 'ul' ? 'ul' : 'ol';
        const listClass = currentListType === 'ul' ? "list-disc" : "list-decimal";
        
        blocks.push(
            <Key key={`list-${blocks.length}`} className={`${listClass} list-outside ml-4 space-y-1 my-2`}>
                {currentListItems.map((item, idx) => (
                    <li key={idx} className="pl-1 leading-relaxed">
                        {item}
                    </li>
                ))}
            </Key>
        );
        currentListItems = [];
        currentListType = null;
    };

    lines.forEach((line, index) => {
        const trimmed = line.trim();
        
        // Empty lines often reset lists or paragraphs
        if (!trimmed) {
            flushList();
            return;
        }

        // Headers
        if (trimmed.startsWith('#')) {
            flushList();
            const level = trimmed.match(/^#+/)?.[0].length || 0;
            const text = trimmed.slice(level).trim();
            const content = parseInline(text);
            
            if (level === 1) blocks.push(<h3 key={`h-${index}`} className="font-bold text-sm mt-4 mb-2 text-slate-900 border-b border-slate-100 pb-1">{content}</h3>);
            else if (level === 2) blocks.push(<h4 key={`h-${index}`} className="font-bold text-xs mt-3 mb-1 uppercase text-slate-700">{content}</h4>);
            else blocks.push(<h5 key={`h-${index}`} className="font-bold text-xs mt-2 text-slate-600">{content}</h5>);
            return;
        }

        // Blockquotes
        if (trimmed.startsWith('> ')) {
            flushList();
            blocks.push(
                <blockquote key={`bq-${index}`} className="border-l-2 border-slate-300 pl-3 py-1 my-2 text-slate-500 italic bg-slate-50/50 rounded-r">
                    {parseInline(trimmed.slice(2))}
                </blockquote>
            );
            return;
        }

        // Unordered List
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
            if (currentListType === 'ol') flushList();
            currentListType = 'ul';
            currentListItems.push(parseInline(trimmed.slice(2)));
            return;
        }

        // Ordered List
        const orderedMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
        if (orderedMatch) {
            if (currentListType === 'ul') flushList();
            currentListType = 'ol';
            currentListItems.push(parseInline(orderedMatch[2]));
            return;
        }

        // Regular Text
        // If we are in a list, maybe this is a continuation?
        // For simplicity, we treat it as a new paragraph if it's not indented.
        // But LLMs often output multi-line list items without indentation.
        // Heuristic: If previous line was list item and this line is text, append to last list item?
        // Let's stick to simple block separation for now. 
        // If user wants multi-line list item, they usually indent. 
        // Without indentation, it's a paragraph.
        flushList();
        blocks.push(
            <div key={`p-${index}`} className="my-1.5 leading-relaxed">
                {parseInline(trimmed)}
            </div>
        );
    });

    flushList(); // Final flush

    return (
        <div className="space-y-1 text-xs text-slate-600 font-sans">
            {blocks}
        </div>
    );
};
