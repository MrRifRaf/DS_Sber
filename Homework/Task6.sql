-- 1. Выберите заказчиков из Германии, Франции и Мадрида, выведите их название, страну и адрес.
SELECT CustomerName, Address, Country
  FROM Customers
 WHERE Country
    IN ('Germany', 'France')
    OR City = 'Madrid';

-- 2. Выберите топ 3 страны по количеству заказчиков, выведите их названия и количество записей.
  SELECT Country, COUNT(CustomerID) AS CustomerCount
    FROM Customers
GROUP BY Country
ORDER BY CustomerCount DESC
   LIMIT 3;

-- 3. Выберите перевозчика, который отправил 10-й по времени заказ, выведите его название, и дату отправления.
  SELECT s.ShipperName, o.OrderDate
    FROM Orders AS o
    JOIN Shippers as s
      ON o.ShipperID = s.ShipperID
ORDER BY o.OrderDate ASC
   LIMIT 1
  OFFSET 9;

-- 4. Выберите самый дорогой заказ, выведите список товаров с их ценами.
  SELECT o.OrderID, p.ProductName, p.Price
    FROM Products AS p
    JOIN OrderDetails AS od
      ON p.ProductID = od.ProductID
    JOIN Orders AS o
      ON o.OrderID = od.OrderID
   WHERE o.OrderID
      IN (SELECT o.OrderID
            FROM Orders AS o
            JOIN OrderDetails AS od
              ON o.OrderID = od.OrderID
            JOIN Products AS p
              ON p.ProductID = od.ProductID
        GROUP BY o.OrderID
        ORDER BY SUM(od.Quantity * p.Price) DESC
           LIMIT 1);

-- 5. Какой товар больше всего заказывали по количеству единиц товара, выведите его название и количество единиц в каждом из заказов.
  SELECT p.ProductName, o.OrderID, od.Quantity
    FROM Products AS p
    JOIN OrderDetails AS od
      ON p.ProductID = od.ProductID
    JOIN Orders AS o
      ON o.OrderID = od.OrderID
   WHERE p.ProductID
      IN (SELECT p.ProductID
            FROM Orders AS o
            JOIN OrderDetails AS od
              ON o.OrderID = od.OrderID
            JOIN Products AS p
              ON p.ProductID = od.ProductID
        GROUP BY p.ProductID
        ORDER BY SUM(od.Quantity) DESC
           LIMIT 1);

-- 6. Выведите топ 5 поставщиков по количеству заказов, выведите их названия, страну, контактное лицо и телефон.
  SELECT s.SupplierName, s.Country, s.ContactName, s.Phone
    FROM Products AS p
    JOIN OrderDetails AS od
      ON p.ProductID = od.ProductID
    JOIN Orders AS o
      ON o.OrderID = od.OrderID
    JOIN Suppliers AS s
      ON p.SupplierID = s.SupplierID
GROUP BY s.SupplierName
ORDER BY COUNT(o.OrderID) DESC
   LIMIT 5;

-- 7. Какую категорию товаров заказывали больше всего по стоимости в Бразилии, выведите страну, название категории и сумму.
  SELECT cus.Country, cat.CategoryName, SUM(od.Quantity * p.Price) AS TotalCost
    FROM Products AS p
    JOIN OrderDetails AS od
      ON p.ProductID = od.ProductID
    JOIN Orders AS o
      ON o.OrderID = od.OrderID
    JOIN Categories AS cat
      ON cat.CategoryID = p.CategoryID
    JOIN Customers AS cus
      ON cus.CustomerID = o.CustomerID
   WHERE cus.Country = 'Brazil'
GROUP BY cat.CategoryName
ORDER BY TotalCost DESC
   LIMIT 1;

-- 8. Какая разница в стоимости между самым дорогим и самым дешевым заказом из США.
  SELECT (MAX(TotalCost) - MIN(TotalCost)) AS CostDiff
    FROM (SELECT o.OrderID, SUM(od.Quantity * p.Price) AS TotalCost
            FROM Products AS p
            JOIN OrderDetails AS od
              ON p.ProductID = od.ProductID
            JOIN Orders AS o
              ON o.OrderID = od.OrderID
            JOIN Customers AS cus
              ON cus.CustomerID = o.CustomerID
           WHERE cus.Country = 'USA'
        GROUP BY o.OrderID);

-- 9. Выведите количество заказов у каждого их трех самых молодых сотрудников, а также имя и фамилию во второй колонке.
  SELECT COUNT(o.OrderID) AS CountOrders, (e.FirstName || ' ' || e.LastName) AS Name
    FROM Orders AS o
    JOIN Employees AS e
      ON o.EmployeeID = e.EmployeeID
  WHERE o.EmployeeID
      IN (SELECT EmployeeID
            FROM Employees
        ORDER BY BirthDate DESC
           LIMIT 3)
GROUP BY Name;

-- 10. Сколько банок крабового мяса всего было заказано.
  SELECT p.ProductName, SUM(od.Quantity * SUBSTR(Unit, 1, 2)) AS Tins
    FROM Products AS p
    JOIN OrderDetails AS od
      ON p.ProductID = od.ProductID
   WHERE LOWER(p.ProductName) LIKE '%crab%meat';
