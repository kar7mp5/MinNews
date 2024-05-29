import React, { useState, useEffect } from 'react';
import axios, { AxiosResponse } from 'axios';

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
                // Fetch metadata from the backend endpoint
                const response: AxiosResponse<Metadata> = await axios.get(`http://localhost:8000/metadata?url=${encodeURIComponent(url)}`);
                const responseData: Metadata = response.data;

                setMetadata(responseData);
            } catch (err) {
                console.error('Error fetching metadata:', err);
                // If an error occurs, set default metadata
                setMetadata({
                    title: url,
                    description: '',
                    image: RequestFailImage,
                    url,
                });
            } finally {
                setLoading(false);
            }
        };

        fetchMetadata();
    }, [url]); // Dependency array with 'url'

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
