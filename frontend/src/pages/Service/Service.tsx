// src/components/Service.tsx
import React, { useState, useEffect } from 'react';
import LinkPreview from 'components/LinkPreview';
import style from 'styles/Main.module.css';

const Service: React.FC = () => {
    const [keyword, setKeyword] = useState<string>(() => sessionStorage.getItem('keyword') || '');
    const [number, setNumber] = useState<number>(() => {
        const storedNumber = sessionStorage.getItem('number');
        return storedNumber ? parseInt(storedNumber, 10) : 5;
    });
    const [urls, setUrls] = useState<string[]>(() => {
        const storedUrls = sessionStorage.getItem('urls');
        return storedUrls ? JSON.parse(storedUrls) : [];
    });
    const [loading, setLoading] = useState<boolean>(false);

    useEffect(() => {
        sessionStorage.setItem('keyword', keyword);
        sessionStorage.setItem('number', number.toString());
        sessionStorage.setItem('urls', JSON.stringify(urls));
    }, [keyword, number, urls]);

    const fetchUrls = async () => {
        try {
            setLoading(true);
            const response = await fetch('http://localhost:8000/get_news', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ keyword, number }),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const data = await response.json();
            setUrls(data.links);
        } catch (error) {
            console.error("Failed to fetch URLs:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault();
        fetchUrls();
    };

    return (
        <main className={style.content}>
            <h1>Service Page</h1>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={keyword}
                    onChange={(e) => setKeyword(e.target.value)}
                    placeholder="Enter keyword"
                    required
                />
                <input
                    type="number"
                    value={number}
                    onChange={(e) => setNumber(parseInt(e.target.value))}
                    placeholder="Number of articles"
                    min="1"
                    required
                />
                <button type="submit">Search</button>
            </form>
            {loading ? (
                <div>Loading...</div>
            ) : (
                urls.map((url, index) => (
                    <LinkPreview key={index} url={url} />
                ))
            )}
        </main>
    );
};

export default Service;
