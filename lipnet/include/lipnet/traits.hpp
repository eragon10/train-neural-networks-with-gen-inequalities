/*
 * Copyright 2020 Niklas Funcke <niklas.funcke@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __TRAITS_HPP__
#define __TRAITS_HPP__

#include <type_traits>
#include <typeindex>
#include <functional>
#include <typeinfo>
#include <string>
#include <cstdarg>
#include <memory>

namespace std {


/**
 * @brief The format function. Like sprintf but with std::string.
 */
template<typename ...ARGS>
inline std::string format(const char* format, ARGS... args)
{
    size_t size = std::snprintf( nullptr, 0, format, std::forward<ARGS>(args)...) + 1;
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf(buf.get(), size, format, args...);
    return std::string(buf.get(), buf.get() + size - 1 );
}

/// void type. Holdes nothing.
struct void_type{};





/**
 * @brief Helper template implementation of a for loop.
 *        With compile type evaluation.
 */
template <auto... Xs, typename F>
constexpr inline void for_values(F&& f)
{
    (f.template operator()<Xs>(), ...);
}


/**
 * @brief Template implementation of a for loop.
 *        With compile type evaluation.
 */
template <auto B, auto E, typename F>
constexpr inline void for_range(F&& f)
{
    using t = std::common_type_t<decltype(B), decltype(E)>;

    [&f]<auto... Xs>(std::integer_sequence<t, Xs...>)
    {
        for_values<(B + Xs)...>(f);
    }
    (std::make_integer_sequence<t, E - B>{});
}
















/************************************************************************************
 * 
 * THE FOLLOWING IST COPIED FROM
 *  https://en.cppreference.com/w/cpp/experimental/is_detected
 * 
 */


namespace detail {
template <class Default, class AlwaysVoid,
          template<class...> class Op, class... Args>
struct detector {
  using value_t = std::false_type;
  using type = Default;
};

template <class Default, template<class...> class Op, class... Args>
struct detector<Default, std::void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
  using type = Op<Args...>;
};

} // namespace detail

struct nonesuch {
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    void operator=(nonesuch const&) = delete;
};

template <template<class...> class Op, class... Args>
using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;

template <template<class...> class Op, class... Args>
using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;

template <class Default, template<class...> class Op, class... Args>
using detected_or = detail::detector<Default, void, Op, Args...>;


template <typename T1, typename T2>
inline size_t constexpr offset_of(T1 T2::*member) {
    constexpr T2 object {};
    return size_t(&(object.*member)) - size_t(&object);
}


/************************************************************************************/

}

#endif // __TRAITS_HPP__
