window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

window.dashAgGridComponentFunctions.PosterWithTooltip = function (props) {
    const title = props.value.title;
    const src = props.value.src;

    const [hovered, setHovered] = React.useState(false);
    const [tooltipStyle, setTooltipStyle] = React.useState({});

    // Event handlers to set and remove the tooltip
    const handleMouseEnter = (e) => {
        setHovered(true);

        // Position the tooltip based on the image's position
        const rect = e.target.getBoundingClientRect();
        setTooltipStyle({
            top: rect.top - 100 + window.scrollY + 'px', // Adjust for scroll position
            left: rect.left + window.scrollX + 30 + 'px',    // Adjust for scroll position
            visibility: 'visible',
        });
    };

    const handleMouseLeave = () => {
        setHovered(false);
    };

    const img = React.createElement("img", {
        src: src,
        title: title,
        onMouseEnter: handleMouseEnter,
        onMouseLeave: handleMouseLeave,
        style: {
            height: "50px", // normal image size
            maxWidth: "100%",
            objectFit: "contain",
            display: "block",
            cursor: "pointer"
        },
        key: "image"
    });

    // Render the tooltip using React.createPortal
    const tooltip = hovered ? ReactDOM.createPortal(
        React.createElement("div", {
            style: {
                ...tooltipStyle,
                position: "absolute",
                zIndex: 1000,  // Ensures it stays above the grid content
                width: "200px",  // Adjust width to your preference
                height: "200px", // Adjust height to your preference
                backgroundColor: "rgba(0,0,0,0.7)",
                borderRadius: "8px",
                padding: "5px",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
            },
            onMouseEnter: () => setHovered(true), // Keep tooltip visible when hovered
            onMouseLeave: handleMouseLeave,      // Hide tooltip when mouse leaves
            key: "tooltip"
        }, React.createElement("img", {
            src: src,
            title: title,
            style: {
                maxWidth: "100%",  // Enlarge the image to fit the tooltip
                maxHeight: "100%",
                objectFit: "contain"
            }
        })),
        document.body // Append tooltip to the body
    ) : null;

    return React.createElement("div", {
        style: {
            display: "flex",
            justifyContent: "flex-start",
            alignItems: "center",
            height: "100%",
            position: "relative",  // Ensure relative positioning for the tooltip
        }
    }, [img, tooltip]);
};


window.dashAgGridComponentFunctions.StarRating = function (props) {
    const rating = parseFloat(props.value) || 0;
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating - fullStars >= 0.25 && rating - fullStars < 0.85;
    const maxStars = 5;

    function getColor(rating) {
        if (rating <= 1.5) return '#ff003e';
        if (rating <= 2.5) return '#ff0060';
        if (rating <= 3.5) return '#d933c2';
        if (rating <= 4.0) return '#b84edd';
        if (rating <= 4.5) return '#8a63f1';
        return '#4073ff';
    }

    const starColor = getColor(rating);
    const stars = [];

    for (let i = 0; i < maxStars; i++) {
        let iconClass = "fa-regular fa-star"; // empty star

        if (i < fullStars) {
            iconClass = "fa-solid fa-star"; // full
        } else if (i === fullStars && hasHalfStar) {
            iconClass = "fa-solid fa-star-half-stroke"; // half
        }

        stars.push(
            React.createElement("i", {
                key: i,
                className: iconClass,
                style: {
                    color: starColor,
                    fontSize: "20px",
                    paddingRight: "2px"
                }
            })
        );
    }

    return React.createElement("div", { 
        style: { 
            display: "flex", 
            alignItems: "center" 
        } 
    }, stars);
};

