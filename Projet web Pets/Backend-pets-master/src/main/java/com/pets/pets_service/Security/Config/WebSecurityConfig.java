package com.pets.pets_service.Security.Config;

import com.pets.pets_service.Service.ClientService;
import com.pets.pets_service.Service.VetService;
import com.pets.pets_service.Service.PPService;

import lombok.AllArgsConstructor;

import java.util.Arrays;

import javax.servlet.http.HttpServletResponse;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.authentication.builders.AuthenticationManagerBuilder;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.web.authentication.AuthenticationFailureHandler;
import org.springframework.security.web.authentication.AuthenticationSuccessHandler;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

@Configuration
@AllArgsConstructor
@EnableWebSecurity
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {

    private final ClientService clientService;
    private final VetService vetService;
    private final PPService ppService;
    private final BCryptPasswordEncoder bCryptPasswordEncoder;

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
                .cors().and()
                .csrf().disable()
                .authorizeRequests()
                .antMatchers("/api/v*/*registration/**")
                .permitAll()
                .anyRequest()
                .authenticated()
                .and()
                .formLogin()
                    .defaultSuccessUrl("/home")
                    .successHandler(loginSuccessHandler())
                    .failureHandler(loginFailureHandler())
                    .permitAll()
                .and()
                .logout()
                    .logoutUrl("/api/v*/logout")
                    .logoutSuccessUrl("http://localhost:4200/")
                    .invalidateHttpSession(true)
                    .deleteCookies("JSESSIONID");
        ;
    }

    @Bean
    public AuthenticationSuccessHandler loginSuccessHandler() {
        return (request, response, authentication) -> {
            // custom logic to modify the response
            response.setStatus(HttpServletResponse.SC_OK);
            response.setContentType("application/json");
            response.getWriter().write("{\"succsess\":true}");
        };
    }

    @Bean
    public AuthenticationFailureHandler loginFailureHandler() {
        return (request, response, exception) -> {
            // custom logic to modify the response
            response.setStatus(HttpServletResponse.SC_OK);
            response.setContentType("application/json");
            response.getWriter().write("{\"succsess\":false}");
        };
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList("http://localhost:4200"));
        configuration.setAllowCredentials(true);
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE"));
        configuration.setAllowedHeaders(Arrays.asList("Content-Type", "Authorization"));

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);

        // add a specific path to the CORS configuration
        CorsConfiguration pathConfig = new CorsConfiguration();
        pathConfig.addAllowedOrigin("http://localhost:4200");
        // pathConfig.addAllowedMethod("GET");
        // pathConfig.addAllowedHeader("Authorization");

        return source;
    }

    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        auth.authenticationProvider(clientDaoAuthenticationProvider());
        auth.authenticationProvider(vetDaoAuthenticationProvider());
        auth.authenticationProvider(ppDaoAuthenticationProvider());
    }

    @Bean
    public DaoAuthenticationProvider ppDaoAuthenticationProvider() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setPasswordEncoder(bCryptPasswordEncoder);
        provider.setUserDetailsService(ppService);
        return provider;
    }

    @Bean
    public DaoAuthenticationProvider clientDaoAuthenticationProvider() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setPasswordEncoder(bCryptPasswordEncoder);
        provider.setUserDetailsService(clientService);
        return provider;
    }

    @Bean
    public DaoAuthenticationProvider vetDaoAuthenticationProvider() {
        DaoAuthenticationProvider provider = new DaoAuthenticationProvider();
        provider.setPasswordEncoder(bCryptPasswordEncoder);
        provider.setUserDetailsService(vetService);
        return provider;
    }
}
