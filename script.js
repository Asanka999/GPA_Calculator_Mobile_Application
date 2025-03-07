
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const coursesContainer = document.getElementById('courses-container');
    const addCourseBtn = document.getElementById('add-course');
    const calculateSemesterBtn = document.getElementById('calculate-semester');
    const calculateOverallBtn = document.getElementById('calculate-overall');
    const clearAllBtn = document.getElementById('clear-all');
    const semesterNameInput = document.getElementById('semester-name');
    const semesterGpaDisplay = document.getElementById('semester-gpa');
    const overallGpaDisplay = document.getElementById('overall-gpa');
    const semestersList = document.getElementById('semesters-list');
   
    // Grade points mapping
    const gradePoints = {
        'A+': 4.0, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'F': 0.0
    };
   
    // Array to store all semesters
    let semesters = loadSemesters() || [];
   
    // Initialize the app
    init();
   
    // Add event listeners
    addCourseBtn.addEventListener('click', addCourse);
    calculateSemesterBtn.addEventListener('click', calculateSemesterGpa);
    calculateOverallBtn.addEventListener('click', calculateOverallGpa);
    clearAllBtn.addEventListener('click', clearAllData);
   
    // Initialize the app
    function init() {
        // Add first empty course row
        if (coursesContainer.children.length === 0) {
            addCourse();
        }
       
        // Render saved semesters
        updateSemestersList();
       
        // Calculate overall GPA from saved data
        calculateOverallGpa();
    }
   
    // Add a new course row
    function addCourse() {
        const courseRow = document.createElement('div');
        courseRow.className = 'course-row';
       
        courseRow.innerHTML = `
            <input type="text" class="course-name" placeholder="Course Name">
            <input type="number" class="course-credits" placeholder="Credits" min="0" step="0.5">
            <select class="course-grade">
                <option value="" disabled selected>Grade</option>
                <option value="A+">A+</option>
                <option value="A">A</option>
                <option value="A-">A-</option>
                <option value="B+">B+</option>
                <option value="B">B</option>
                <option value="B-">B-</option>
                <option value="C+">C+</option>
                <option value="C">C</option>
                <option value="C-">C-</option>
                <option value="D+">D+</option>
                <option value="D">D</option>
                <option value="F">F</option>
            </select>
            <button class="remove-course">Remove</button>
        `;
       
        // Add event listener to remove button
        const removeBtn = courseRow.querySelector('.remove-course');
        removeBtn.addEventListener('click', () => {
            courseRow.remove();
        });
       
        coursesContainer.appendChild(courseRow);
    }
   
    // Calculate semester GPA
    function calculateSemesterGpa() {
        const courseRows = coursesContainer.querySelectorAll('.course-row');
        const semesterName = semesterNameInput.value.trim() || `Semester ${semesters.length + 1}`;
       
        let totalCredits = 0;
        let totalPoints = 0;
        let courses = [];
       
        // Calculate GPA for current semester
        courseRows.forEach(row => {
            const courseName = row.querySelector('.course-name').value;
            const creditsValue = row.querySelector('.course-credits').value;
            const grade = row.querySelector('.course-grade').value;
           
            if (courseName && creditsValue && grade) {
                const credits = parseFloat(creditsValue);
                const points = gradePoints[grade] * credits;
               
                totalCredits += credits;
                totalPoints += points;
               
                courses.push({
                    name: courseName,
                    credits: credits,
                    grade: grade
                });
            }
        });
       
        // Calculate semester GPA
        const semesterGpa = totalCredits > 0 ? (totalPoints / totalCredits).toFixed(2) : '0.00';
        semesterGpaDisplay.textContent = semesterGpa;
       
        // Save semester data
        if (courses.length > 0) {
            const semester = {
                name: semesterName,
                gpa: parseFloat(semesterGpa),
                totalCredits: totalCredits,
                totalPoints: totalPoints,
                courses: courses,
                date: new Date().toISOString()
            };
           
            // Check if semester with same name exists and replace it
            const existingIndex = semesters.findIndex(s => s.name === semesterName);
            if (existingIndex !== -1) {
                semesters[existingIndex] = semester;
            } else {
                semesters.push(semester);
            }
           
            // Save and update UI
            saveSemesters();
            updateSemestersList();
            calculateOverallGpa();
        }
    }
   
    // Calculate overall GPA
    function calculateOverallGpa() {
        if (semesters.length === 0) {
            overallGpaDisplay.textContent = '0.00';
            return;
        }
       
        let totalCredits = 0;
        let totalPoints = 0;
       
        semesters.forEach(semester => {
            totalCredits += semester.totalCredits;
            totalPoints += semester.totalPoints;
        });
       
        const overallGpa = totalCredits > 0 ? (totalPoints / totalCredits).toFixed(2) : '0.00';
        overallGpaDisplay.textContent = overallGpa;
    }
   
    // Update semesters list in the UI
    function updateSemestersList() {
        semestersList.innerHTML = '';
       
        if (semesters.length === 0) {
            semestersList.innerHTML = `
                <div class="empty-state">
                    <p>No semesters added yet</p>
                    <p>Complete a semester calculation to see it here</p>
                </div>
            `;
            return;
        }
       
        // Sort semesters by date (newest first)
        semesters.sort((a, b) => new Date(b.date) - new Date(a.date));
       
        semesters.forEach(semester => {
            const semesterItem = document.createElement('div');
            semesterItem.className = 'semester-item';
           
            semesterItem.innerHTML = `
                <div>
                    <div class="semester-name">${semester.name}</div>
                    <div class="credits-summary">${semester.totalCredits} credits</div>
                </div>
                <div class="semester-gpa">GPA: ${semester.gpa.toFixed(2)}</div>
            `;
           
            // Add click event to load semester
            semesterItem.addEventListener('click', () => {
                loadSemester(semester);
            });
           
            semestersList.appendChild(semesterItem);
        });
    }
   
    // Load a semester into the edit area
    function loadSemester(semester) {
        // Set semester name
        semesterNameInput.value = semester.name;
       
        // Clear current courses
        coursesContainer.innerHTML = '';
       
        // Add courses from the semester
        semester.courses.forEach(course => {
            const courseRow = document.createElement('div');
            courseRow.className = 'course-row';
           
            courseRow.innerHTML = `
                <input type="text" class="course-name" placeholder="Course Name" value="${course.name}">
                <input type="number" class="course-credits" placeholder="Credits" min="0" step="0.5" value="${course.credits}">
                <select class="course-grade">
                    <option value="" disabled>Grade</option>
                    <option value="A+" ${course.grade === 'A+' ? 'selected' : ''}>A+</option>
                    <option value="A" ${course.grade === 'A' ? 'selected' : ''}>A</option>
                    <option value="A-" ${course.grade === 'A-' ? 'selected' : ''}>A-</option>
                    <option value="B+" ${course.grade === 'B+' ? 'selected' : ''}>B+</option>
                    <option value="B" ${course.grade === 'B' ? 'selected' : ''}>B</option>
                    <option value="B-" ${course.grade === 'B-' ? 'selected' : ''}>B-</option>
                    <option value="C+" ${course.grade === 'C+' ? 'selected' : ''}>C+</option>
                    <option value="C" ${course.grade === 'C' ? 'selected' : ''}>C</option>
                    <option value="C-" ${course.grade === 'C-' ? 'selected' : ''}>C-</option>
                    <option value="D+" ${course.grade === 'D+' ? 'selected' : ''}>D+</option>
                    <option value="D" ${course.grade === 'D' ? 'selected' : ''}>D</option>
                    <option value="F" ${course.grade === 'F' ? 'selected' : ''}>F</option>
                </select>
                <button class="remove-course">Remove</button>
            `;
           
            // Add event listener to remove button
            const removeBtn = courseRow.querySelector('.remove-course');
            removeBtn.addEventListener('click', () => {
                courseRow.remove();
            });
           
            coursesContainer.appendChild(courseRow);
        });
       
        // Update semester GPA display
        semesterGpaDisplay.textContent = semester.gpa.toFixed(2);
    }
   
    // Clear all data
    function clearAllData() {
        if (confirm('Are you sure you want to clear all your GPA data? This cannot be undone.')) {
            // Clear semesters array
            semesters = [];
           
            // Clear localStorage
            localStorage.removeItem('gpaSemesters');
           
            // Clear UI
            coursesContainer.innerHTML = '';
            addCourse(); // Add one empty course row
            semesterNameInput.value = '';
            semesterGpaDisplay.textContent = '0.00';
            overallGpaDisplay.textContent = '0.00';
           
            // Update UI
            updateSemestersList();
        }
    }
   
    // Save semesters to localStorage
    function saveSemesters() {
        localStorage.setItem('gpaSemesters', JSON.stringify(semesters));
    }
   
    // Load semesters from localStorage
    function loadSemesters() {
        const saved = localStorage.getItem('gpaSemesters');
        return saved ? JSON.parse(saved) : null;
    }
});