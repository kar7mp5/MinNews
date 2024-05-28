import React, { useEffect, useState } from 'react';
import axios from 'axios';

import style from 'styles/LinkPreview.module.css';
import RequestFailImage from 'assets/Request_Fail.png'; // Import the Request Fail image

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

    useEffect(() => {
        const fetchMetadata = async () => {
            setLoading(true);

            try {
                // axios 요청을 보내서 메타데이터 가져오기
                const response = await axios.get(`https://api.microlink.io?url=${encodeURIComponent(url)}`);
                const { data } = response;

                if (!data || !data.title || !data.description || !data.image) {
                    throw new Error('Failed to fetch complete metadata');
                }

                setMetadata({
                    title: data.title,
                    description: data.description,
                    image: data.image.url,
                    url,
                });
            } catch (err: unknown) {
                console.error('Error fetching metadata:', err);
                setMetadata({
                    title: url,
                    description: '',
                    image: RequestFailImage, // Use the Request Fail image
                    url,
                });
            } finally {
                setLoading(false);
            }
        };

        fetchMetadata();
    }, [url]);

    if (loading) return <div className={style.loading}>Loading...</div>;

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
