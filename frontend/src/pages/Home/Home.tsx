// Home.tsx
import React from 'react';

// contents
import Problem from 'pages/Home/Problem';

import style from 'styles/Main.module.css';

const Home: React.FC = () => {
    return (
        <main className={style.content}>
            <h1>Project Introduction</h1>
            <Problem />
        </main>
    );
};

export default Home;
