// MarkdownRenderer.tsx
import React from 'react';
import MarkdownPreview from '@uiw/react-markdown-preview';
import style from 'styles/MarkdownRenderer.module.css';

interface MarkdownProps {
    markdownContent: string;
}

const MarkdownRenderer: React.FC<MarkdownProps> = ({ markdownContent }) => {
    return (
        <div>
            <MarkdownPreview className={style.markdown} source={markdownContent} />
        </div>
    );
};

export default MarkdownRenderer;
