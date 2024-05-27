import React, { useEffect, useState } from 'react';
import axios from 'axios';

import style from 'styles/LinkPreview.module.css';

interface LinkPreviewProps {
    url: string;
}

interface Metadata {
    title: string;
    description: string;
    image: string;
    url: string;
}

const LinkPreview: React.FC<LinkPreviewProps> = ({ url }) => {
    const [metadata, setMetadata] = useState<Metadata | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchMetadata = async () => {
            setLoading(true);
            setError(null);

            try {
                const response = await axios.get(`https://api.microlink.io?url=${encodeURIComponent(url)}`);
                const { title, description, image } = response.data.data;

                if (!title || !description || !image?.url) {
                    setError('Failed to fetch complete metadata');
                } else {
                    setMetadata({
                        title,
                        description,
                        image: image.url,
                        url,
                    });
                }
            } catch (err) {
                console.error('Error fetching metadata:', err);
                setError('Failed to fetch metadata');
            } finally {
                setLoading(false);
            }
        };

        fetchMetadata();
    }, [url]);

    if (loading) return <div>Loading...</div>;
    if (error) return <div>{error}</div>;

    return (
        metadata && (
            <div className={style.link_preview}>
                <div>
                    <img className={style.img_container} src={metadata.image} alt={metadata.title} />
                </div>
                <div>
                    <a href={metadata.url} target="_blank" rel="noopener noreferrer">
                        {metadata.title}
                    </a>
                    <p>{metadata.description}</p>
                </div>
            </div>
        )
    );
};

export default LinkPreview;
