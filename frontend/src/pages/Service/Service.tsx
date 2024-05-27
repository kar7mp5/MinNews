// src/components/Service.tsx
import React from 'react';
import LinkPreview from 'components/LinkPreview';

import style from 'styles/Main.module.css';

const Service: React.FC = () => {
    return (
        <main className={style.content}>
            <h1>Service Page</h1>
            <LinkPreview url="https://www.theatlantic.com/ideas/archive/2024/05/qassem-soleimani-iran-middle-east/678472/" />
            <LinkPreview url="https://www.youtube.com/watch?v=Tj0WQnwUInc" />
        </main>
    );
};

export default Service;
