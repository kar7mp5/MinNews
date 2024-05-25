import React from 'react';
import { FaBars } from 'react-icons/fa';
import styles from 'styles/Header.module.css';

interface HeaderProps {
    isOpen: boolean;
    toggleSidebar: () => void;
}

const Header: React.FC<HeaderProps> = ({ isOpen, toggleSidebar }) => {
    return (
        <header className={styles.header}>
            <button className={styles.toggleButton} onClick={toggleSidebar}>
                {isOpen ? <> </> : <FaBars className={styles.sidebarToggle} />}
            </button>
            <h1 className={isOpen ? styles.headerTextOpen : ''}>
                My Simple Homepage
            </h1>
        </header>
    );
};

export default Header;
