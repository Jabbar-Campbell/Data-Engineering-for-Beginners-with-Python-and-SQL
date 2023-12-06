--CREATE TABLE users (user_id int, age int);
--INSERT INTO users (user_id, age)
--VALUES (1, 25),(2, 30),(3, 22),(4, 35),(5, 28),(6, 40),(7, 19),(8, 27),(9, 33),(10, 29),(11, 31),(12, 24),(13, 38),(14, 26),(15, 23),(16, 32),(17, 21),(18, 37),
--(19, 34),(20, 36),(21, 20),(22, 42),(23, 45),(24, 18),(25, 39),(26, 43),(27, 48),(28, 50),(29, 44),(30, 52),(31, 47),(32, 19),(33, 55),(34, 58),(35, 21),(36, 60),
--(37, 56),(38, 22),(39, 62),(40, 65),(41, 59),(42, 23),(43, 68),(44, 70),(45, 64),(46, 24),(47, 72),(48, 75),(49, 69),(50, 26)

-- You have a table called "users" (see above), with columns "user_id" and "age". Return the "count of users" that are aged above 50


---------------------------------------------------------------------------------------
SELECT COUNT(age) AS count_of_users --count the different ages and rename
FROM users
WHERE age > 50 
-----------------------------------------------------------------------------------




-- You have orders table with columns order_id, client_id, order_amount . 
-- Output the average order amount and the average order count for each client.'
--------------------------------------------------------------------------------- 
SELECT client_id, --select a column 
COUNT(*) AS orders, --count the instances and rename
AVG(order_amount) AS avg_order -- compute the average of a column and rename

FROM orders
GROUP BY client_id
----------------------------------------------------------------------------------
 



--Write a SQL SELECT query that retrieves the following information:
--Department name
--Average salary for employees in that department
--Group the results by department name.
--Sort the results in ascending order based on the department name.
--------------------------------------------------------------------------------
SELECT department AS DEPARMENT ,  -- we can select and rename
salary as Average_Salary -- we can select and rename columns off the bat

FROM employees
GROUP BY department 
ORDER BY department  asc  -- set the order
--------------------------------------------------------------------------------



--Write a SQL SELECT query that retrieves the following information:
--Product category
--Total sales amount for each product category
--Average sales amount for each product category
--Group the results by product category.
--Sort the results in descending order based on total sales amount
--------------------------------------------------------------------------------
SELECT  
product_category AS "Product Category",--we simply rename
SUM(sales_amount) AS "Total Sales Amount" --we calculate and rename
AVG(sales_amount) AS "Average Sales Amount" --we calculate and rename

FROM sales
GROUP BY product_category
ORDER BY "Total Sales Amount" DESC



--                                     SUBQUERIES AND JOINING
-- THE WITH STATMENT aka Common Table Expression (cte) allows us to chain queries
-- it like creating a primary Data table thats been manipulated
-- that you can then select from to make things less confusing 
-- at allows us to chain selects 
------------------------------------------------------------------------------------
-WITH 
cte_name AS ( 
    SELECT column_1, column_2,
                     AVG(...) AS...
                     SUM(...) AS...
                     FROM my_table
                     GROUP BY ...
                     ORDER BY ...
),


cte2_name AS (
    SELECT column_1, column_2,
                     AVG(...) AS...
                     SUM(...) AS...
                     FROM my_table
                     GROUP BY ...
                     ORDER BY ...
)

--Main query that references columns from both CTE's
SELECT cte.column_name....,cte_2.column_name...
FROM cte_name cte_1 -- set an alias for first query   
JOIN cte2_name cte2; -- set an alias for second query
------------------------------------------------------------------------------- 




--Write a SQL SELECT query that retrieves the following information:
--Department name
--Employee with the highest salary in each department (full name)
--Highest salary in each department
--Employee with the lowest salary in each department (full name)
--The lowest salary in each department
--Group the results by department name.
--Sort the results in ascending order based on the department name.
--------------------------------------------------------------------------------------
WITH --allows us to chain querys and name them
HighestSalaries AS (
    SELECT
        department AS "Department",
        first_name || ' ' || last_name AS "Highest Earner",
        MAX(salary) AS "Highest Salary"
    FROM
        employees
    GROUP BY
        department
),

LowestSalaries AS (
    SELECT
        department AS "Department",
        first_name || ' ' || last_name AS "Lowest Earner", --concats to columns into one
        MIN(salary) AS "Lowest Salary"
    FROM
        employees
    GROUP BY
        department
)


SELECT h.Highest Earner h.Higest Salary l.Lowest Earner l.Lowest Salary

FROM HighestSalaries h --sql allows alias or short hand 
JOIN LowestSalaries l ON e.Department = d.Department; --sql allows alias or short hand so we distingish above
-----------------------------------------------------------------------------------------