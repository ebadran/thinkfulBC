--  Convert currency columns to float and date columns to date type:
CREATE TABLE cp_listings
AS SELECT * FROM listings;

CREATE TABLE cp_calendar
AS SELECT * FROM calendar;

UPDATE cp_calendar SET price=SUBSTR(price, 2) WHERE price LIKE '$%';
UPDATE cp_calendar SET price=CAST(price AS float);
UPDATE cp_calendar SET date=date(date);

UPDATE cp_listings SET price=SUBSTR(price, 2) WHERE price LIKE '$%';
UPDATE cp_listings SET price=CAST(price AS float);

-- Q1: What's the most expensive listing? What else can you tell me about the listing?
-- A: There are six most expensive listings, all of them apartments, and each worth $999 a day.
-- One of them is located in Brooklyn (Clinton Hill), the other five are located in Manhattan.
-- The listing in Clinton Hill has the most reviews (32 reviews and a rating of 93).
SELECT
	price,
	id,
	neighbourhood_cleansed,
	neighbourhood_group_cleansed,
	property_type,
	CAST(availability_365 AS float) listing_availability,
	number_of_reviews,
	review_scores_rating
FROM cp_listings
GROUP BY id
ORDER BY price DESC;


-- Q2: What neighborhoods seem to be the most popular?
-- A1: The four neighborhoods with the most listings are Williamsburg (3920 listings),
-- Bedford-Stuyvesant (3178), Harlem (2623), and Bushwick (2203).
SELECT
	neighbourhood_cleansed,
	COUNT(neighbourhood_cleansed) listings_per_neighbourhood
FROM cp_listings
GROUP BY 1
ORDER BY listings_per_neighbourhood DESC;

-- A2: A neighborhood can have a lot of listings, but not too many guests. The neighborhoods
-- with the most listings that have received at least three reviews per month are
-- Bedford-Stuyvesant	(451), Williamsburg	(372), Hell's Kitchen	(365), and Harlem	(335).
WITH
most_reviewed AS (
	SELECT
		id,
		neighbourhood_cleansed,
		CAST(reviews_per_month AS float) monthly_reviews
	FROM cp_listings
	WHERE monthly_reviews >=3
	GROUP BY 1
	)
SELECT
	neighbourhood_cleansed,
	COUNT(neighbourhood_cleansed) count_neighbourhood
FROM most_reviewed
GROUP BY 1
ORDER BY count_neighbourhood DESC;


-- Q3: What time of year is the cheapest time to go to your city? What about the busiest?
-- A: Eight out of the top 14 days in which listings had their maximum prices fall within the New Year's
-- Holidays (from Dec/30th to Jan/06th) or the Easter Holidays (March/30th and 31st).
-- Another six days (out of the top 14 days) fall within the second semester (from July to October).
-- The bottom 14 days in which listings had their minimum prices fall within the winter season, from
-- January 22 to March 09.
WITH
max_prices AS (
	SELECT
		MAX(price) max_price,
		DATE(date) dates,
		listing_id
	FROM cp_calendar
	GROUP BY 3
	)
SELECT
	COUNT(max_price),
	dates
FROM max_prices
GROUP BY dates
ORDER BY 1 DESC;
