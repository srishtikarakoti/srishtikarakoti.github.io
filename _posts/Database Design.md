---
title: "NJIT Database Design"
date: 2020-06-15
tags: [java, database design, sql]
header:
  image: 
excerpt: "Java, Database Design, SQL"
mathjax: "true"
---


## Goals
New Jersey Institute of Technology is starting to move some of its applications into a database environment.

The goal is to set-up a database system which fulfills all the requirements of New Jersey Institute of Technology.

 
## Requirements
A requirements analysis was conducted which has identified a number of things about the operations and goals of New Jersey Institute of Technology. 

We have adhered to these requirements in order to achieve a richer design.

New Jersey Institute of Technology keeps records about a number of items. These are the following:

1.	First and foremost, it keeps track of its students who are uniquely identified by their Student Ids (different from the SSN). For each student, his/her SSN, address, high school, major (identified by a particular department) and year are other essential information.
2.	Information about staff is also maintained. The name and SSN (which uniquely identify each staff) are essential information for each staff. The address, salary are other required information.
3.	A particular type of staff is the faculty. In addition to the general staff information, the rank and course load (in terms of maximum number of courses that the particular faculty can teach) information are stored for each faculty.
4.	A faculty may be assigned to multiple departments (joint appointments). This enables them to teach courses at multiple departments as long as the total number of courses is not higher than the course load.
5.	Information about each department is maintained. Each department is identified by a unique codes. The department name, the location of the main office, latest annual budget are the information that are stored. Each department has a chairperson who is a faculty member.
6.	Physical room facilities of the university are identified. The identification is by means of a building code and room number. Other information include the audio-visual equipment and the capacity of the room.
7.	Departments offer a number of courses. Course information includes the course code, course name, course credit (using the nomenclature that is followed at NJIT). No two courses in the university can have the same code. For each course, the teaching assistant requirements (in terms of number of hours per week) are also stored.
8.	Sections of courses are identified by a number. Note that there cannot exist a section unless it is related to an existing course. For the sections, New Jersey Institute of Technology keeps information about the room, the weekdays (1 or 2) and the time the lectures are held and the maximum enrollment.
9.	Each section is taught by one faculty member. Faculty members can teach multiple sections of a course or sections of different courses.
10.	Students register for a number of course sections but to only one section of the same course.
11.	For each course a number of teaching assistants (TA) can be assigned. Teaching assistants are students who have been hired as staff. Full-time TAs are allowed to work a maximum of 20 hours a week while half-time TAs can work 12 hours a week.

 
## Overview
Database design involves organizing data in such a way that it can be accessed using queries. The organization will depend on the purpose of the database. The data should be arranged into tables and columns. Table relationships will also need to be defined. Primary keys will need to be specified in the design. Finally, the normalization will have to be applied to the design. In our case, the data we used is typical of many universities where student and course information is collected for use by the university. Common uses for this information are student identification, student schedules, building information, and course information. Information is also filtered to produce specific data at a given time.
	To make use of the university data, we created two applications. One application allows students to register for courses and the other will be used by the university for class and student information. We found that designing the applications using Java will work best for our purposes. From our ER diagram, we were able to code the relationships between the different sets of data. One of the challenges is to link our java applications to the sql server. With the link created, we were able to find student information by entering the primary keys into the application. We were also able to generate class information by entering the section number. 

## Entity Relationship

The challenge in designing an application and a database is to accurately link the relationships together. The following relationships needed to be established in our case: a belong to relationship between faculty and department, a chair relationship between faculty and department, a teach relationship between faculty and section, an assign relationship between teaching assistant and section, a register relationship between student and section, a CS relationship between course and section, a SR relationship between room and section, a possess relationship between room and building, a locate relationship between the department and building, a DC relationship between Department and Course, and a major relationship between department and section. In our design, the strong entities were student, staff, departments, courses and building. The weak entities were section and room. Specialization included faculty and teaching assistant.

### Entities
●	Staff

●	Department

●	Building

●	Student

●	Course
 
### Weak Entities
●	Section depends on Course

●	Room depends on Building
 
### Specialization
●	Faculty is a specialization of Staff

●	Teaching Assistant is a specialization of both Staff and student
 
### Relationships
●	Belong to ( between Faculty and Department)

●	Chair ( between Faculty and Department)

●	Teach( between Faculty and Section)

●	Assign(between Teaching Assistant and Section

●	Register( between Student and Section)

●	CS( between Course and Section)

●	SR( between Room and Section)

●	Possess( between Room and Building)

●	Locate (between Department and Building)

●	DC (between Department and Course)

●	Major ( between Department and Student)


### Summary
The database design involved creating many tables. Our first table used a constraint assignment 1, foreign key T_SSN, and reference was staff T_SSN, constraint assignment 2, foreign key course_no, and reference, course_no, and constraint assignment 2 foreign key sec_no, and section sec_no. Our second table contained building_id not null auto_increcement, b_name, default null, and location default null where the primary key was listed as building_id. We also had to alter our second table to include values main campus and city center. The third table has course_no not null, course_name default null, credit double default null, TA_hr_req default null, dept_code default null. The primary key was listed as the course_no. The other key was dept_code and the ference was departments. We altered the table to include courses accounting, english as language, english literature, financial reporting, introduction to java, marketing, marketing in the 21st century and python basics. The fourth table included dept_code not null, dept_name default null, a_budget default null, building_id default null, dept_chair default null, and dept_location default null. The primary key was listed as dept_code. Constraint 1 included foreign  key Building_id and references building and constraint 2 had foreign key dept_chair and references staff. The fifth table included departments business management, computer science, finance and accounts, and language arts. The sixth table listed faculty_department where the key where t_ssn and dept_code. The seventh table contained registration information such as s_id, sec_no, and course_no. References for this table included student, section and courses. The eight table contained building information including building_id, room_no, capacity and audio_visual. The ninth table contained section information including sec_no, couse_no, c_year, semester max_enroll, and instructor_ssn where the keys were course_no and instructor_no. The tenth table listed the section and room information including building_id, room_no, course_no, sec_no, weekday and time which referenced room, courses and section. The eleventh table contained staff information which included t_ssn, t_name, t_add, staff_function, t_salary, staff_rank, course_load and work_hr. The twelfth table contained student information s_id, s_ssn, s_name, s_add, s_high, s_year and major where the primary key s_id. Student names used included Adam, Britney, Cristina, David, Emma and Frank.

Step 1 - Strong Entities: all the strong entities are mapped into tables
Student(S_ID, S_SSN, S_Name, S_Add, S_High, S_Year)  S_ID is a primary key and S_SSN is a candidate key
Staff(T_SSN, T_Name, T_ADD, T_Salary)
Departments(Dept_code, Dept_Name, A_Budget, T_SSN, Building_Id)
Courses(Course_No, Course_Name, Credit, TA_hr_req)
Building(Building_ID, B_Name, Location)
 
Step 2 - Weak Entities: all the weak entities are mapped into tables with a composite key that includes the key of the strong entities they depend on
Section(Sec_No, Course_No, Year, Semester, T_SSN, Max_enroll)
Room( Building_Id, Room_No, Capacity, Audio_Visual)
 
Step 3 Specialization
Solution 1: Map the specialized entities as relations:
Faculty(T_SSN, Rank, Course_Load)
TeachingAssistant(T_SSN, T_SID, Work_hr)
This solution will save space but will require joining staff and either of the tables Faculty and TeachingAssistant to get the full information about a faculty member or a teaching assistant
 
Solution 2: Move the specialized attributes into Staff
Staff(T_SSN, T_Name, T_ADD, function, T_Salary, Rank, Course_Load, Work_hr)
The attribute function is introduced to distinguish the staff members. Its value will be “faculty”, “TA” and any other function that can be found among the staff members. This solution will waste space (for example the attribute work_hr is not needed for a faculty) but will be fast in accessing staff information (all the information about a staff is in one single tuple)
 
The 2 solutions are about a tradeoff between time and space. I prefer solution 2 because space is cheap and solution 2 will be faster when accessing staff information
 
Step 4 - 1:1 Relationship
Chair: Department has a total participation into the relationship whereas faculty (now staff) has a partial participation. In other words, every department has a chair and not all the faculty members are chairs. So the relationship chair is mapped as a foreign key in Department
 
Departments(Dept_code, Dept_Name, A_Budget, T_SSN, Building_Id, Dept_chair)
 
Step 5 - 1:N Relationship
Major: Total participation form student(N side of the relationship). So major is mapped as a foreign key in student
Student(S_ID, S_SSN, S_Name, S_Add, S_High, S_Year, Major) 
 
If the student had a partial participation (let’s say we can have undeclared student) then the best mapping would be to create a table
Major(S_ID, Dept_code)
 
DC: Total participation form Course (N side of the relationship). So DC is mapped as a foreign key in Course
Courses(Course_No, Course_Name, Credit, TA_hr_req, Dept_code )
 
Locate: Total participation form Department (N side of the relationship). So Locate is mapped as a foreign key in Department
Departments(Dept_code, Dept_Name, A_Budget, T_SSN, Building_Id, Dept_chair, Dept_location)
 
Teach: Total participation form Section (N side of the relationship). So Teach is mapped as a foreign key in Section
Section(Sec_No, Course_No, Year, Semester T_SSN, Max_enroll, Instructor_SSN)
 
Step 6 - M:N Relationship: all the M:N relationships are mapped into tables

Belongs to:
Faculty_Department(T_SSN, Dept_code)
 
Assign:
Assignment(T_SSN, Course_No , Sec_No)
 
Register:
Registration(S_ID, Sec_No, Course_No)
SR:
SectionInRoom(Building_ID, Room_No, Course_No, Sec_No, Weekday, Time)

## Final Schema
Staff(T_SSN, T_Name, T_ADD, function, T_Salary, Rank, Course_Load, Work_hr)

Departments(Dept_code, Dept_Name, A_Budget, T_SSN, Building_Id, Dept_chair, Dept_location)

Student(S_ID, S_SSN, S_Name, S_Add, S_High, S_Year, Major) 

Courses(Course_No, Course_Name, Credit, TA_hr_req, Dept_code )

Building(Building_ID, B_Name, Location)

Section(Sec_No, Course_No, Year, Semester T_SSN, Max_enroll, Instructor_SSN)

Room( Building_Id, Room_No, Capacity, Audio_Visual)

SectionInRoom(Building_ID, Room_No, Course_No, Sec_No, Weekday, Time)

Registration(S_ID, Sec_No, Course_No)

Assignment(T_SSN, Course_No , Sec_No)

Faculty_Department(T_SSN, Dept_code)

## Application Program Design

We used java language to design our application. In our design, the user was given a choice of two processes including student registration and class list generation. By entering the student id on the student registration application, the student name was displayed as well as the course name, semester and year. If the student attempts to register for the same course twice, an error message will appear stating that there is a duplicate entry. Additionally, the application will display an error message if the course is too full stating that maximum enrollment was reached. By entering the section number in the class list generation application, the course number is displayed as well as the course name, section number, year, semester, room number, weekday, time and instructor name. By pressing enter without entering a section number, the entire class list is displayed.

 

## Database Design



```python

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

-- Dumping structure for table 80.assignment
DROP TABLE IF EXISTS `assignment`;
CREATE TABLE IF NOT EXISTS `assignment` (
  `T_SSN` varchar(20) NOT NULL,
  `Course_No` varchar(20) NOT NULL,
  `Sec_No` varchar(20) NOT NULL,
  PRIMARY KEY (`T_SSN`,`Course_No`,`Sec_No`),
  KEY `Course_No` (`Course_No`),
  KEY `Sec_No` (`Sec_No`),
  CONSTRAINT `assignment_ibfk_1` FOREIGN KEY (`T_SSN`) REFERENCES `staff` (`T_SSN`),
  CONSTRAINT `assignment_ibfk_2` FOREIGN KEY (`Course_No`) REFERENCES `courses` (`Course_No`),
  CONSTRAINT `assignment_ibfk_3` FOREIGN KEY (`Sec_No`) REFERENCES `section` (`Sec_No`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.assignment: ~0 rows (approximately)
/*!40000 ALTER TABLE `assignment` DISABLE KEYS */;
/*!40000 ALTER TABLE `assignment` ENABLE KEYS */;

-- Dumping structure for table 80.building
DROP TABLE IF EXISTS `building`;
CREATE TABLE IF NOT EXISTS `building` (
  `Building_Id` int(11) NOT NULL AUTO_INCREMENT,
  `B_Name` varchar(100) DEFAULT NULL,
  `Location` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`Building_Id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=latin1;

-- Dumping data for table 80.building: ~1 rows (approximately)
/*!40000 ALTER TABLE `building` DISABLE KEYS */;
INSERT INTO `building` (`Building_Id`, `B_Name`, `Location`) VALUES
	(1, 'Main Campus', 'City Center');
/*!40000 ALTER TABLE `building` ENABLE KEYS */;

-- Dumping structure for table 80.courses
DROP TABLE IF EXISTS `courses`;
CREATE TABLE IF NOT EXISTS `courses` (
  `Course_No` varchar(20) NOT NULL,
  `Course_Name` varchar(100) DEFAULT NULL,
  `Credit` double DEFAULT NULL,
  `TA_hr_req` varchar(100) DEFAULT NULL,
  `Dept_Code` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`Course_No`),
  KEY `Dept_Code` (`Dept_Code`),
  CONSTRAINT `courses_ibfk_1` FOREIGN KEY (`Dept_Code`) REFERENCES `departments` (`Dept_Code`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.courses: ~8 rows (approximately)
/*!40000 ALTER TABLE `courses` DISABLE KEYS */;
INSERT INTO `courses` (`Course_No`, `Course_Name`, `Credit`, `TA_hr_req`, `Dept_Code`) VALUES
	('ACCT101', 'Accounting', 4, NULL, 'FA'),
	('ENG101', 'Enghlish as Language', 3, NULL, 'LA'),
	('ENG102', 'English Literature', 4, NULL, 'LA'),
	('FIN101', 'Financial Reporting', 4, NULL, 'FA'),
	('JAVA101', 'Introduction to Java', 4, NULL, 'CS'),
	('MKT101', 'Marketing', 4, NULL, 'BM'),
	('MKT102', 'Marketing in 21st Century', 4, NULL, 'BM'),
	('PY101', 'Python Basics', 4, NULL, 'CS');
/*!40000 ALTER TABLE `courses` ENABLE KEYS */;

-- Dumping structure for table 80.departments
DROP TABLE IF EXISTS `departments`;
CREATE TABLE IF NOT EXISTS `departments` (
  `Dept_Code` varchar(20) NOT NULL,
  `Dept_Name` varchar(100) DEFAULT NULL,
  `A_Budget` int(11) DEFAULT NULL,
  `Building_Id` int(11) DEFAULT NULL,
  `Dept_Chair` varchar(20) DEFAULT NULL,
  `Dept_location` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`Dept_Code`),
  KEY `Building_Id` (`Building_Id`),
  KEY `Dept_Chair` (`Dept_Chair`),
  CONSTRAINT `departments_ibfk_1` FOREIGN KEY (`Building_Id`) REFERENCES `building` (`Building_Id`),
  CONSTRAINT `departments_ibfk_2` FOREIGN KEY (`Dept_Chair`) REFERENCES `staff` (`T_SSN`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.departments: ~4 rows (approximately)
/*!40000 ALTER TABLE `departments` DISABLE KEYS */;
INSERT INTO `departments` (`Dept_Code`, `Dept_Name`, `A_Budget`, `Building_Id`, `Dept_Chair`, `Dept_location`) VALUES
	('BM', 'Business Management', NULL, 1, '102', NULL),
	('CS', 'Computer Sceinces', NULL, 1, '100', NULL),
	('FA', 'Finance and Accounts', NULL, 1, '103', NULL),
	('LA', 'Language Arts', NULL, 1, '101', NULL);
/*!40000 ALTER TABLE `departments` ENABLE KEYS */;

-- Dumping structure for table 80.faculty_department
DROP TABLE IF EXISTS `faculty_department`;
CREATE TABLE IF NOT EXISTS `faculty_department` (
  `T_SSN` varchar(20) NOT NULL,
  `Dept_Code` varchar(20) NOT NULL,
  PRIMARY KEY (`T_SSN`,`Dept_Code`),
  KEY `Dept_Code` (`Dept_Code`),
  CONSTRAINT `faculty_department_ibfk_1` FOREIGN KEY (`T_SSN`) REFERENCES `staff` (`T_SSN`),
  CONSTRAINT `faculty_department_ibfk_2` FOREIGN KEY (`Dept_Code`) REFERENCES `departments` (`Dept_Code`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.faculty_department: ~0 rows (approximately)
/*!40000 ALTER TABLE `faculty_department` DISABLE KEYS */;
/*!40000 ALTER TABLE `faculty_department` ENABLE KEYS */;

-- Dumping structure for table 80.registration
DROP TABLE IF EXISTS `registration`;
CREATE TABLE IF NOT EXISTS `registration` (
  `S_ID` int(11) NOT NULL,
  `Sec_No` varchar(20) NOT NULL,
  `Course_No` varchar(20) NOT NULL,
  PRIMARY KEY (`S_ID`,`Sec_No`,`Course_No`),
  KEY `Sec_No` (`Sec_No`),
  KEY `Course_No` (`Course_No`),
  CONSTRAINT `registration_ibfk_1` FOREIGN KEY (`S_ID`) REFERENCES `student` (`S_ID`),
  CONSTRAINT `registration_ibfk_2` FOREIGN KEY (`Sec_No`) REFERENCES `section` (`Sec_No`),
  CONSTRAINT `registration_ibfk_3` FOREIGN KEY (`Course_No`) REFERENCES `courses` (`Course_No`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.registration: ~12 rows (approximately)
/*!40000 ALTER TABLE `registration` DISABLE KEYS */;
INSERT INTO `registration` (`S_ID`, `Sec_No`, `Course_No`) VALUES
	(1, 'ACA', 'ACCT101'),
	(1, 'BMA', 'MKT101'),
	(1, 'LAA', 'ENG101'),
	(1, 'LAB', 'ENG102'),
	(2, 'BMB', 'MKT102'),
	(2, 'CSA', 'JAVA101'),
	(2, 'CSB', 'PY101'),
	(3, 'BMB', 'MKT102'),
	(3, 'CSA', 'JAVA101'),
	(4, 'BMA', 'MKT101'),
	(4, 'BMB', 'MKT102'),
	(6, 'BMA', 'MKT101');
/*!40000 ALTER TABLE `registration` ENABLE KEYS */;

-- Dumping structure for table 80.room
DROP TABLE IF EXISTS `room`;
CREATE TABLE IF NOT EXISTS `room` (
  `Building_Id` int(11) NOT NULL,
  `Room_No` varchar(20) NOT NULL,
  `Capacity` int(11) DEFAULT NULL,
  `Audio_Visual` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`Building_Id`,`Room_No`),
  CONSTRAINT `room_ibfk_1` FOREIGN KEY (`Building_Id`) REFERENCES `building` (`Building_Id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.room: ~2 rows (approximately)
/*!40000 ALTER TABLE `room` DISABLE KEYS */;
INSERT INTO `room` (`Building_Id`, `Room_No`, `Capacity`, `Audio_Visual`) VALUES
	(1, 'A1', 100, 'Multimedia, Speakers'),
	(1, 'A2', 100, 'Multimedia, Speakers');
/*!40000 ALTER TABLE `room` ENABLE KEYS */;

-- Dumping structure for table 80.section
DROP TABLE IF EXISTS `section`;
CREATE TABLE IF NOT EXISTS `section` (
  `Sec_No` varchar(20) NOT NULL,
  `Course_No` varchar(20) DEFAULT NULL,
  `C_Year` int(11) DEFAULT NULL,
  `Semester` varchar(100) DEFAULT NULL,
  `Max_enroll` int(11) DEFAULT NULL,
  `Instructor_SSN` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`Sec_No`),
  KEY `Course_No` (`Course_No`),
  KEY `Instructor_SSN` (`Instructor_SSN`),
  CONSTRAINT `section_ibfk_1` FOREIGN KEY (`Course_No`) REFERENCES `courses` (`Course_No`),
  CONSTRAINT `section_ibfk_2` FOREIGN KEY (`Instructor_SSN`) REFERENCES `staff` (`T_SSN`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.section: ~8 rows (approximately)
/*!40000 ALTER TABLE `section` DISABLE KEYS */;
INSERT INTO `section` (`Sec_No`, `Course_No`, `C_Year`, `Semester`, `Max_enroll`, `Instructor_SSN`) VALUES
	('ACA', 'ACCT101', 2019, 'Spring', 10, '101'),
	('ACB', 'FIN101', 2019, 'Autumn', 10, '101'),
	('BMA', 'MKT101', 2019, 'Spring', 10, '103'),
	('BMB', 'MKT102', 2019, 'Autumn', 3, '103'),
	('CSA', 'JAVA101', 2019, 'Spring', 10, '100'),
	('CSB', 'PY101', 2019, 'Autumn', 10, '100'),
	('LAA', 'ENG101', 2019, 'Spring', 10, '102'),
	('LAB', 'ENG102', 2019, 'Autumn', 10, '102');
/*!40000 ALTER TABLE `section` ENABLE KEYS */;

-- Dumping structure for table 80.sectioninroom
DROP TABLE IF EXISTS `sectioninroom`;
CREATE TABLE IF NOT EXISTS `sectioninroom` (
  `Building_Id` int(11) NOT NULL,
  `Room_No` varchar(20) NOT NULL,
  `Course_No` varchar(20) NOT NULL,
  `Sec_No` varchar(20) NOT NULL,
  `Weekday` varchar(20) DEFAULT NULL,
  `Time` time DEFAULT NULL,
  PRIMARY KEY (`Building_Id`,`Room_No`,`Course_No`,`Sec_No`),
  KEY `Course_No` (`Course_No`),
  KEY `Sec_No` (`Sec_No`),
  CONSTRAINT `sectioninroom_ibfk_1` FOREIGN KEY (`Building_Id`, `Room_No`) REFERENCES `room` (`Building_Id`, `Room_No`),
  CONSTRAINT `sectioninroom_ibfk_2` FOREIGN KEY (`Course_No`) REFERENCES `courses` (`Course_No`),
  CONSTRAINT `sectioninroom_ibfk_3` FOREIGN KEY (`Sec_No`) REFERENCES `section` (`Sec_No`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.sectioninroom: ~8 rows (approximately)
/*!40000 ALTER TABLE `sectioninroom` DISABLE KEYS */;
INSERT INTO `sectioninroom` (`Building_Id`, `Room_No`, `Course_No`, `Sec_No`, `Weekday`, `Time`) VALUES
	(1, 'A1', 'ACCT101', 'ACA', 'Monday', '08:30:00'),
	(1, 'A1', 'ENG101', 'LAA', 'Tuesday', '11:00:00'),
	(1, 'A1', 'ENG102', 'LAB', 'Wednesday', '12:30:00'),
	(1, 'A1', 'MKT101', 'BMA', 'Thursday', '15:00:00'),
	(1, 'A2', 'FIN101', 'ACB', 'Tuesday', '11:00:00'),
	(1, 'A2', 'JAVA101', 'CSA', 'Monday', '08:30:00'),
	(1, 'A2', 'MKT102', 'BMB', 'Thursday', '15:00:00'),
	(1, 'A2', 'PY101', 'CSB', 'Wednesday', '12:30:00');
/*!40000 ALTER TABLE `sectioninroom` ENABLE KEYS */;

-- Dumping structure for table 80.staff
DROP TABLE IF EXISTS `staff`;
CREATE TABLE IF NOT EXISTS `staff` (
  `T_SSN` varchar(20) NOT NULL,
  `T_NAME` varchar(100) DEFAULT NULL,
  `T_ADD` varchar(250) DEFAULT NULL,
  `staff_function` varchar(100) DEFAULT NULL,
  `T_Salary` int(11) DEFAULT NULL,
  `staff_rank` varchar(100) DEFAULT NULL,
  `Course_Load` int(11) DEFAULT NULL,
  `Work_hr` int(11) DEFAULT NULL,
  PRIMARY KEY (`T_SSN`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

-- Dumping data for table 80.staff: ~6 rows (approximately)
/*!40000 ALTER TABLE `staff` DISABLE KEYS */;
INSERT INTO `staff` (`T_SSN`, `T_NAME`, `T_ADD`, `staff_function`, `T_Salary`, `staff_rank`, `Course_Load`, `Work_hr`) VALUES
	('100', 'Prof Albert', NULL, NULL, NULL, NULL, NULL, NULL),
	('101', 'Prof Brian', NULL, NULL, NULL, NULL, NULL, NULL),
	('102', 'Prof Charlie', NULL, NULL, NULL, NULL, NULL, NULL),
	('103', 'Prof Douglas', NULL, NULL, NULL, NULL, NULL, NULL),
	('104', 'Prof Elrado', NULL, NULL, NULL, NULL, NULL, NULL),
	('105', 'Prof Franklin', NULL, NULL, NULL, NULL, NULL, NULL);
/*!40000 ALTER TABLE `staff` ENABLE KEYS */;

-- Dumping structure for table 80.student
DROP TABLE IF EXISTS `student`;
CREATE TABLE IF NOT EXISTS `student` (
  `S_ID` int(11) NOT NULL AUTO_INCREMENT,
  `S_SSN` varchar(20) DEFAULT NULL,
  `S_Name` varchar(100) DEFAULT NULL,
  `S_Add` varchar(250) DEFAULT NULL,
  `S_High` varchar(100) DEFAULT NULL,
  `S_Year` int(11) DEFAULT NULL,
  `Major` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`S_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=latin1;

-- Dumping data for table 80.student: ~6 rows (approximately)
/*!40000 ALTER TABLE `student` DISABLE KEYS */;
INSERT INTO `student` (`S_ID`, `S_SSN`, `S_Name`, `S_Add`, `S_High`, `S_Year`, `Major`) VALUES
	(1, '1', 'Adam', 'Adam Street', 'A', 2019, 'Artificial Intelligence'),
	(2, '2', 'Britney', 'Britney Street', 'B', 2019, 'Business Administration'),
	(3, '3', 'Christina', 'Christy Street', 'C', 2019, 'Computer Sciences'),
	(4, '4', 'David', 'David Street', 'D', 2019, 'Databases'),
	(5, '5', 'Emaa', 'Emaa Street', 'E', 2019, 'English Literature'),
	(6, '6', 'Frank', 'Frank Street', 'F', 2019, 'Financial Accounting');
/*!40000 ALTER TABLE `student` ENABLE KEYS */;

/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;

```

## Applications Design


```python
import java.util.Scanner;

public class Main {
    private static DatabaseHelper db;
    private static Scanner input = new Scanner(System.in);

    public static void main(String[] args) {
        db = new DatabaseHelper();
        String choice = "";

        while (!choice.equals("3")) {

            System.out.println("");
            System.out.println("Available Applications");
            System.out.println("*".repeat(40));
            System.out.print("1: Student Registration \n2: Class List Generation \n3: End Program\nSelect Application: ");
            choice = input.nextLine().strip();
    
            switch (choice) {
                case "1":
                    registerStudent();
                    break;
                case "2":
                    generateClassList();
                    break;
                case "3":
                    System.out.println("Bye!");
                    break;
            }
        }
    }

    private static void generateClassList() {
        System.out.print("Enter Section No (Press Enter for All): ");
        String section_no = input.nextLine().strip().toUpperCase();
        db.generateClassList(section_no);
    }

    private static void registerStudent() {

        System.out.println("\nRegister Student");
        System.out.println("-".repeat(40));

        int s_id = 0;
        while (s_id<=0) {
            try {
                System.out.print("Enter Student ID: ");
                s_id = Integer.parseInt(input.nextLine().strip());
            } catch (NumberFormatException e) {
                System.out.println("Invalid Student ID. Must be a number");
                s_id = 0;
            }
        }

        String s_name = db.getStudent(s_id);
        if (s_name.contains("Error")) return;

        System.out.print("Enter Section No: ");
        String section_no = input.nextLine().strip().toUpperCase();
        if (!db.checkVacancy(section_no)) return;

        String course_no = db.getCourseBySectionNo(section_no);
        if (course_no.contains("Error")) return;

        db.addRegistration(s_id, section_no, course_no);

    }
```

## Java to SQL Link


```python
import java.io.BufferedReader;
import java.io.FileReader;
import java.sql.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DatabaseHelper {
    protected Connection connection;

    final String USER = "root";
    final String PASS = "";
    final String DATABASE = "80";

    public DatabaseHelper() {
        try {
            Class.forName("com.mysql.jdbc.Driver");
            this.connection = DriverManager.getConnection(
                    "jdbc:mysql://localhost:3306/" + DATABASE, USER, PASS);
        } catch (ClassNotFoundException | SQLException e) {
            e.printStackTrace();
        }
    }

    public String getStudent(int s_id){
        String sql = "", response = "";
        ResultSet resultSet = null;
        try {
            sql = "select s_name, s_ssn from student where s_id = " + s_id;
            Statement statement = this.connection.createStatement();
            statement.execute(sql);
            resultSet = statement.getResultSet();
            if(resultSet.next()) {
                response = resultSet.getString(1) + ", " + resultSet.getString(2);
            } else {
                response = "Error: Student record not found";
            }
        } catch (SQLException throwables) {
            response =  throwables.getMessage();
        }
        System.out.println(">> " + response);
        return response;
    }

    public String getCourse(String course_no){
        String sql = "", response = "";
        ResultSet resultSet = null;
        try {
            sql = "select * from courses where course_no = '" + course_no + "'";
            Statement statement = this.connection.createStatement();
            statement.execute(sql);
            resultSet = statement.getResultSet();
            if(resultSet.next()) {
                response = resultSet.getString(1);
            } else {
                response = "Error: Course  not found";
            }
        } catch (SQLException throwables) {
            response =  throwables.getMessage();
        }
        System.out.println(response);
        return response;
    }

    public boolean checkVacancy(String section_no){
        ResultSet resultSet;
        boolean response;

        String sql = "SELECT MAX(section.Max_enroll) Capacity, COUNT(registration.S_ID) Enrollment\n" +
                "FROM section left join registration ON section.Sec_No = registration.Sec_No\n" +
                "WHERE section.sec_no = '" + section_no + "'";

        try {
            Statement statement = this.connection.createStatement();
            statement.execute(sql);
            resultSet = statement.getResultSet();
            resultSet.next();
            int capacity = resultSet.getInt(1);
            int enrollment = resultSet.getInt(2);
            response = capacity > enrollment;
            if (!response)
                System.out.println(">> Maximum Enrollment of " + enrollment + " Reached.");
        } catch (SQLException throwables) {
            response = false;
            System.out.println(">> " + throwables.getMessage());
        }
        return response;
    }

    public String getCourseBySectionNo(String section_no){
        ResultSet resultSet = null;
        String response, displayText;
        String sql = "select course_no, c_year, semester from section where sec_no = '" + section_no + "'";

        try {
            Statement statement = this.connection.createStatement();
            statement.execute(sql);
            resultSet = statement.getResultSet();
            if(resultSet.next()) {
                displayText = resultSet.getString(1) + ", " + resultSet.getInt(2) + " - " + resultSet.getString(3);
                response = resultSet.getString(1);
            } else {
                response = "Error: Section not found";
                displayText = response;
            }
        } catch (SQLException throwables) {
            response =  throwables.getMessage();
            displayText = response;
        }
        System.out.println(">> " + displayText);
        return response;
    }

    public void addRegistration(int s_id, String sec_no, String course_no) {
        String response = "";

        String sql = "INSERT INTO registration (s_id, sec_no, course_no) VALUES ";
        sql += "(" + s_id + ", '" + sec_no + "', '" + course_no + "')";

        try {
            PreparedStatement preparedStatement =  connection.prepareStatement(sql);
            int numRows = preparedStatement.executeUpdate(sql);
            response  = "Student Registered successfully.";
        } catch (SQLException throwables) {
            response = "Error while Student Registration " + "\n\r" + throwables.getMessage();
        }
        System.out.println(">> " + response);
    }

    public void generateClassList(String section_no){
        ResultSet resultSet;
        String response = "";

        String sql = "SELECT\n" +
                "\tcourses.Course_No,\n" +
                "\tcourses.Course_Name,\n" +
                "\tsection.Sec_No,\n" +
                "\tsection.C_Year,\n" +
                "\tsection.Semester,\n" +
                "\tsectioninroom.Room_No,\n" +
                "\tsectioninroom.Weekday,\n" +
                "\tsectioninroom.Time,\n" +
                "\tstaff.T_NAME\n" +
                "FROM \n" +
                "\tsection LEFT JOIN \n" +
                "\tcourses ON section.Course_No = courses.Course_No LEFT JOIN \n" +
                "\tsectioninroom ON section.Sec_No = sectioninroom.Sec_No LEFT JOIN\n" +
                "\tstaff ON section.Instructor_SSN = staff.T_SSN";

        if (section_no != "")
            sql += " WHERE section.sec_no = '" + section_no + "'";

        try {
            Statement statement = this.connection.createStatement();
            statement.execute(sql);
            resultSet = statement.getResultSet();
        } catch (SQLException throwables) {
            System.out.println ("Error while generating class list");
            return;
        }
        String[] cols = {"Course_No", "Course_Name","Sec_No","Year","Semester","Room_No","Weekday","Time","Instructor"};
        String formatStr = " %-10s %-30s %-10s %-10s %-10s %-10s %-12s %-12s %-20s";
        System.out.println("");
        while(true) {
            try {
                if (!resultSet.next()) break;
                ResultSetMetaData resultSetMetaData = resultSet.getMetaData();
                int columnsNumber = resultSetMetaData.getColumnCount();
                String[] format = formatStr.split(" ");
                for (int i = 1; i <= columnsNumber; i++) {
                    System.out.println(String.format("%-15s : %-25s", cols[i-1], resultSet.getString(i)));
                }
                listStudentsBySection(resultSet.getString(3));
            } catch (SQLException throwables) {
                throwables.printStackTrace();
            }
        }
    }

    public void listStudentsBySection(String sec_no){
        ResultSet resultSet;
        String response = "";

        String sql = "SELECT student.S_ID, S_SSN, S_Name, S_Year, Major FROM registration LEFT JOIN student ON registration.S_ID = student.S_ID" +
                "\nWHERE registration.Sec_No = '" + sec_no + "'" +
                "\nORDER BY S_Name";

        try {
            Statement statement = this.connection.createStatement();
            statement.execute(sql);
            resultSet = statement.getResultSet();
        } catch (SQLException throwables) {
            System.out.println ("Error while generating class list");
            return;
        }

        String formatStr = " %-10s %-10s %-20s %-10s %-20s";
        while(true) {
            try {
                if (!resultSet.next()) break;
                ResultSetMetaData resultSetMetaData = resultSet.getMetaData();
                int columnsNumber = resultSetMetaData.getColumnCount();
                String[] format = formatStr.split(" ");
                for (int i = 1; i <= columnsNumber; i++) {
                    response += String.format(format[i], resultSet.getString(i));
                }
                response += "\n\r";
            } catch (SQLException throwables) {
                throwables.printStackTrace();
            }
        }
        String columnNames = String.format(formatStr.replaceAll(" ",""),
                "S_ID", "S_SSN","S_Name","Year","Major");
        columnNames += "\n" + "-".repeat(80) + "\n";
        if (response.length()>5) {
            response = "\n" + columnNames + response;
            System.out.println(response);
        } else {
            System.out.println("\n>> No Registration for this Section\n");
        }
    }

}
```
