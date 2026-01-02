function adjustContentHeight() {
    const navbar = document.querySelector('.navbar-glass');
    const footer = document.querySelector('.footer');
    const contentArea = document.querySelector('.content-area');

    if (contentArea && navbar && footer) {
        const windowHeight = window.innerHeight;
        const navbarHeight = navbar.offsetHeight;
        const footerHeight = footer.offsetHeight;
        const padding = 40; // Additional padding

        const contentHeight = windowHeight - navbarHeight - footerHeight - padding;
        contentArea.style.minHeight = `${Math.max(contentHeight, 300)}px`;
    }
}

// Run on load and resize
window.addEventListener('load', adjustContentHeight);
window.addEventListener('resize', adjustContentHeight);
document.addEventListener('DOMContentLoaded', adjustContentHeight);