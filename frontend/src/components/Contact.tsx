// src/components/Contact.tsx
import React from 'react';
import MarkdownRenderer from 'components/MarkdownRenderer';

const Contact: React.FC = () => {
    return (
        <section id="contact">
            <MarkdownRenderer
                markdownContent={`
## Contact Us

현재 작동되는 프로그램 깃허브 주소입니다.

[깃허브 주소](https://github.com/kimminsum/MinGPT)

제작한 Python 라이브러리 입니다.

[Pypi 사이트 주소](https://pypi.org/project/korean-news-scraper/)  
[Python 라이브러리 깃허브](https://github.com/kimminsum/korean-news-scraper)
            `}
            />
        </section>
    );
};

export default Contact;
