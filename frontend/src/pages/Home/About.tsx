// src/components/About.tsx
import React from 'react';
import MarkdownRenderer from 'components/MarkdownRenderer';

const About: React.FC = () => {
    return (
        <section id="about">
            <MarkdownRenderer
                markdownContent={`
## About Me
현재 데이터분석기초 수강 중인 인하대학교 컴퓨터공학과 12234073 김민섭입니다.
            `}
            />
        </section>
    );
};

export default About;
