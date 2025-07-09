$(function() {

 // --- Модальное окно и формы ---

var modal = $('#loginModal'); 
$('#loginBtn').on('click', function() {
modal.show();
}); 
$('.close').on('click', function() {
modal.hide();
}); 
$(window).on('click', function(event) {
if ($(event.target).is(modal)) {
modal.hide();
}
});
// Переключение форм
$('#showRegister').on('click', function(e) {
e.preventDefault();
$('#loginFormContainer').hide();
$('#registrationFormContainer').show();
$('#showLogin').removeClass('active');
$('#showRegister').addClass('active');
});
$('#showLogin').on('click', function(e) {
e.preventDefault();
$('#registrationFormContainer').hide();
$('#loginFormContainer').show();
$('#showRegister').removeClass('active');
$('#showLogin').addClass('active');
});
// Отправка формы входа
$('#loginForm').on('submit', function(e) {
e.preventDefault();
$.ajax({
type: "POST",
url: loginUrl,

data: $(this).serialize(),
success: function(response) {
window.location.reload();
},
error: function() {
alert('Ошибка входа!');
}
});
});
// Отправка формы регистрации
$('#registrationForm').on('submit', function(e) {
e.preventDefault();
$.ajax({
type: "POST",
url: $(this).attr('action'),
data: $(this).serialize(),
success: function() {
alert('Регистрация успешна!');
window.location.reload();
},
error: function(response) {
var errorMsg = 'Ошибка регистрации!';
if (response.responseJSON && response.responseJSON.error) {
errorMsg = response.responseJSON.error;
}
alert(errorMsg);
}
});
});

 


// --- Автодополнение и поиск --- 
    const input = $('#search-input');
    const resultsBox = $('#autocomplete-results');
    const searchForm = input.closest('form');

    // Получаем CSRF-токен из cookie  
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let cookie of cookies) {
                cookie = cookie.trim();
                if (cookie.startsWith(name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    const csrftoken = getCookie('csrftoken');

    // При фокусе показываем историю запросов
    input.on('focus', function() {
        if ($(this).val().trim() !== '') {
            resultsBox.empty().hide();
            return;
        }
        fetch('/api/get_user_search_history/', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'X-CSRFToken': csrftoken
            },
            credentials: 'include'
        })
        .then(res => res.json())
        .then(history => {
            if (!Array.isArray(history) || history.length === 0) {
                resultsBox.empty().hide();
                return;
            }
            resultsBox.empty();
            history.slice(0, 5).forEach(item => {
                $('<div>')
                    .addClass('autocomplete-item')
                    .text(item.query_text)
                    .appendTo(resultsBox)
                    .on('click', function() {
                        input.val(item.query_text);
                        resultsBox.empty().hide();
                        searchForm.submit();
                    });
            });
            resultsBox.show();
        })
        .catch(() => {
            resultsBox.empty().hide();
        });
    });

    // При вводе показываем автодополнение  
    input.on('input', function() {
        let query = $(this).val().trim();

        if (query.length === 0) {
            input.trigger('focus');
            return;
        }
        if (query.length < 2) {
            resultsBox.empty().hide();
            return;
        }

        $.getJSON('/autocomplete/', { q: query }, function(data) {
    resultsBox.empty();
    if (data.length) {
        resultsBox.show();
        const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`(${escapedQuery})`, 'gi');

        data.forEach(item => {
            const highlighted = item.title.replace(regex, '<strong>\$1</strong>');
            $('<div>')
                .addClass('autocomplete-item')
                .html(highlighted)
                .appendTo(resultsBox)
                .on('click', function() {
                    // При клике сразу переходим на страницу книги
                    window.location.href = '/books/' + item.id + '/';
                });
        });
    } else {
        resultsBox.hide();
    }
});

    });
 
    $(document).on('click', function(e) {
        if (!$(e.target).closest('.search-bar').length) {
            resultsBox.empty().hide();
        }
    });

    // Отправка поискового запроса на сервер для записи в историю
    async function sendSearchQuery(query) {
        if (!query || !query.trim()) return;
        try {
            const response = await fetch('/api/record_user_search/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrftoken,
                },
                body: JSON.stringify({ query_text: query.trim() }),
                credentials: 'include'
            });
            if (!response.ok) {
                const data = await response.json();
                console.error('Error recording search:', data);
            }
        } catch (error) {
            console.error('Network error:', error);
        }
    }

    // При отправке формы — сохраняем запрос в историю и отправляем на сервер
    searchForm.on('submit', function(e) {
        e.preventDefault();
        const query = input.val().trim();
        if (!query) return;

        sendSearchQuery(query);

        this.submit();
    });
});
