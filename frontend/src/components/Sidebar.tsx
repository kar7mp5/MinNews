import React from 'react';
import styles from 'styles/Sidebar.module.css';
import { FaTimes } from 'react-icons/fa';
import { Link } from 'react-router-dom';


interface SidebarProps {
    isOpen: boolean;
    toggleSidebar: () => void;
}


const Sidebar: React.FC<SidebarProps> = ({ isOpen, toggleSidebar }) => {
    return (
        <aside className={`${styles.sidebar} ${isOpen ? styles.open : ''}`}>
            <button className={styles.toggleButton} onClick={toggleSidebar}>
                <FaTimes className={styles.sidebarToggle} />
            </button>
            <nav>
                <ul className={styles.menu}>
                    <li>
                        <Link to='/'>Home</Link>
                    </li>
                    <li>
                        <Link to='/service'>Service</Link>
                    </li>
                    <li>
                        <a href="#problem">Problem</a>
                    </li>
                    <li>
                        <a href="#about">About</a>
                    </li>
                    <li>
                        <a href="#contact">Contact</a>
                    </li>
                </ul>
            </nav>
        </aside>
    );
};

export default Sidebar;
